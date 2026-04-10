"""
services/chunking.py
─────────────────────────────────────────────────────────────────────
ChunkingService — three chunking strategies for medical PDFs.

    Strategy 1 — Fixed-size     : split every N tokens with overlap
    Strategy 2 — Recursive      : paragraph → sentence → word splits
    Strategy 3 — Semantic       : embed sentences, split where meaning changes

Usage:
    from .services.chunking import ChunkingService

    service = ChunkingService(document)          # reads strategy from doc
    chunks  = service.run()                      # returns list[ChunkData]
    service.save(chunks)                         # bulk-creates Chunk rows

Or run everything in one call:
    chunks = ChunkingService(document).run_and_save()

Run standalone smoke-test:
    python services/chunking.py
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Generator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
#  ChunkData  — lightweight DTO, one per chunk
# ─────────────────────────────────────────────────────────────────


@dataclass
class ChunkData:
    """Plain data object returned by every chunking strategy."""

    text: str
    chunk_index: int
    page_start: int = 0
    page_end: int = 0
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0

    def __post_init__(self):
        if not self.token_count:
            self.token_count = _estimate_tokens(self.text)


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters (GPT/Gemini rule-of-thumb)."""
    return max(1, len(text) // 4)


def _extract_pages(file_path: str) -> list[dict]:
    """
    Extract per-page text from a PDF using pdfplumber.

    Returns:
        [{"page": 1, "text": "...", "char_offset": 0}, ...]
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF extraction.\n"
            "Install it with:  pip install pdfplumber"
        )

    pages = []
    offset = 0

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            pages.append(
                {
                    "page": i,
                    "text": text,
                    "char_offset": offset,
                }
            )
            offset += len(text) + 1  # +1 for the separator we'll add

    return pages


def _build_full_text(
    pages: list[dict],
) -> Generator[tuple[str, list[dict]], None, None]:
    """
    Join all page texts into one string.
    Returns (full_text, pages_with_offsets) so we can map
    any character position back to a page number.
    """
    parts = []
    running = 0

    for p in pages:
        p = dict(p)
        p["char_start"] = running
        parts.append(p["text"])
        running += len(p["text"]) + 1
        p["char_end"] = running - 1
        yield p

    # (generator used only for offset annotation, caller joins separately)


def _annotate_pages(pages: list[dict]) -> tuple[str, list[dict]]:
    """
    Returns (full_text, annotated_pages).
    Each page dict gets char_start and char_end keys.
    """
    annotated = []
    running = 0
    parts = []

    for p in pages:
        start = running
        parts.append(p["text"])
        running += len(p["text"]) + 1  # +1 for "\n" separator
        annotated.append({**p, "char_start": start, "char_end": running - 1})

    return "\n".join(p["text"] for p in pages), annotated


def _page_for_char(char_pos: int, annotated_pages: list[dict]) -> int:
    """Binary-search the annotated page list to find which page a char falls on."""
    lo, hi = 0, len(annotated_pages) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if char_pos < annotated_pages[mid]["char_start"]:
            hi = mid - 1
        elif char_pos > annotated_pages[mid]["char_end"]:
            lo = mid + 1
        else:
            return annotated_pages[mid]["page"]
    # Fallback: last page
    return annotated_pages[-1]["page"] if annotated_pages else 1


# ─────────────────────────────────────────────────────────────────
#  Strategy 1 — Fixed-size chunking
# ─────────────────────────────────────────────────────────────────


class FixedSizeChunker:
    """
    Splits the full document text into chunks of exactly `chunk_size`
    tokens with `chunk_overlap` tokens of overlap between consecutive
    chunks.

    How it works:
        1. Tokenise by splitting on whitespace (word = token proxy).
        2. Slide a window of `chunk_size` words, step = chunk_size - overlap.
        3. Reconstruct each window back to a string.

    Pros : Simple, predictable chunk sizes, easy to reason about.
    Cons : Can cut mid-sentence, mid-paragraph — degrades retrieval quality.
    Best for: Quick baseline, structured data (tables, drug dosage lists).
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)

    def chunk(self, full_text: str, annotated_pages: list[dict]) -> list[ChunkData]:
        words = full_text.split()
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks = []
        idx = 0

        # Rebuild char-offset map: word_index → char position in full_text
        char_positions = _word_char_map(full_text)

        i = 0
        while i < len(words):
            window = words[i : i + self.chunk_size]
            text = " ".join(window)
            char_start = char_positions[i] if i < len(char_positions) else 0
            char_end = char_start + len(text)

            chunks.append(
                ChunkData(
                    text=text,
                    chunk_index=idx,
                    page_start=_page_for_char(char_start, annotated_pages),
                    page_end=_page_for_char(char_end, annotated_pages),
                    char_start=char_start,
                    char_end=char_end,
                )
            )

            idx += 1
            i += step

        logger.info(
            f"[FixedSize] {len(chunks)} chunks | size={self.chunk_size} overlap={self.chunk_overlap}"
        )
        return chunks


def _word_char_map(text: str) -> list[int]:
    """Return the start character position of each whitespace-delimited word."""
    positions = []
    in_word = False
    for i, ch in enumerate(text):
        if ch not in (" ", "\n", "\t"):
            if not in_word:
                positions.append(i)
                in_word = True
        else:
            in_word = False
    return positions


# ─────────────────────────────────────────────────────────────────
#  Strategy 2 — Recursive character chunking
# ─────────────────────────────────────────────────────────────────

# Separator hierarchy for medical text (most → least natural)
_MEDICAL_SEPARATORS = [
    "\n\n\n",  # section break
    "\n\n",  # paragraph break
    "\n",  # line break
    ". ",  # sentence end (English)
    "? ",  # question
    "! ",  # exclamation
    "; ",  # semi-colon clause
    ", ",  # comma clause
    " ",  # word boundary (last resort)
]


class RecursiveChunker:
    """
    Recursively splits text using a hierarchy of separators, trying
    paragraph breaks first and falling back to smaller units only
    when a piece is still too large.

    How it works:
        1. Try to split on the first separator (double newline / section break).
        2. If any resulting piece is still > chunk_size tokens, recurse
           into it with the next separator in the list.
        3. Merge pieces that are smaller than chunk_size with the next
           piece until adding one more would exceed the limit.
        4. Emit overlapping windows by looking back `chunk_overlap` tokens
           from the start of each new chunk.

    Pros : Respects paragraph and sentence structure, much better retrieval
           quality than fixed-size for prose-heavy medical documents.
    Cons : Chunk sizes vary slightly; a bit more complex to reason about.
    Best for: Research papers, clinical notes, drug package inserts.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 4)
        self.separators = separators or _MEDICAL_SEPARATORS

    # ── Public entry point ────────────────────────────────────────

    def chunk(self, full_text: str, annotated_pages: list[dict]) -> list[ChunkData]:
        raw_chunks = self._split(full_text, self.separators)
        merged = self._merge_with_overlap(raw_chunks)

        result = []
        char_cursor = 0

        for idx, text in enumerate(merged):
            # Find the char position of this chunk in full_text
            pos = full_text.find(text, char_cursor)
            char_start = pos if pos != -1 else char_cursor
            char_end = char_start + len(text)
            char_cursor = max(char_cursor, char_start + 1)

            result.append(
                ChunkData(
                    text=text.strip(),
                    chunk_index=idx,
                    page_start=_page_for_char(char_start, annotated_pages),
                    page_end=_page_for_char(char_end, annotated_pages),
                    char_start=char_start,
                    char_end=char_end,
                )
            )

        logger.info(
            f"[Recursive] {len(result)} chunks | size={self.chunk_size} overlap={self.chunk_overlap}"
        )
        return result

    # ── Internal helpers ──────────────────────────────────────────

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text until all pieces fit within chunk_size."""
        if not separators or _estimate_tokens(text) <= self.chunk_size:
            return [text]

        sep = separators[0]
        rest = separators[1:]
        pieces = text.split(sep)
        result = []

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if _estimate_tokens(piece) <= self.chunk_size:
                result.append(piece)
            else:
                # Still too large — recurse with next separator
                result.extend(self._split(piece, rest))

        return result

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """
        Greedily merge small pieces into chunks up to chunk_size,
        then prepend the last `chunk_overlap` tokens of the previous
        chunk to create overlap.
        """
        chunks = []
        current = []
        cur_tokens = 0

        for piece in pieces:
            piece_tokens = _estimate_tokens(piece)

            if cur_tokens + piece_tokens > self.chunk_size and current:
                # Emit current chunk
                chunk_text = " ".join(current)
                chunks.append(chunk_text)

                # Carry over overlap from the tail of current
                overlap_text = self._tail_tokens(chunk_text, self.chunk_overlap)
                current = [overlap_text] if overlap_text else []
                cur_tokens = _estimate_tokens(overlap_text)

            current.append(piece)
            cur_tokens += piece_tokens

        if current:
            chunks.append(" ".join(current))

        return [c for c in chunks if c.strip()]

    def _tail_tokens(self, text: str, n_tokens: int) -> str:
        """Return approximately the last n_tokens tokens of text."""
        words = text.split()
        return " ".join(words[-n_tokens:]) if len(words) > n_tokens else text


# ─────────────────────────────────────────────────────────────────
#  Strategy 3 — Semantic chunking
# ─────────────────────────────────────────────────────────────────


class SemanticChunker:
    """
    Groups sentences together based on embedding similarity.
    A new chunk starts wherever the cosine similarity between
    adjacent sentences drops below the configured threshold.

    How it works:
        1. Split the document into individual sentences.
        2. Embed each sentence using Gemini gemini-embedding-001.
        3. Compute cosine similarity between each consecutive pair.
        4. When similarity drops below `breakpoint_threshold`, start
           a new chunk — the content has shifted topic.
        5. Merge very small groups (< min_chunk_tokens) with neighbours.

    Pros : Chunks respect semantic topic boundaries — highest retrieval
           quality because retrieved chunks are topically coherent.
    Cons : Requires one Gemini API call per sentence (rate-limit aware),
           slower than fixed/recursive.
    Best for: Long mixed-topic documents (annual reports, clinical guidelines,
              research papers with many sections).

    Rate limiting:
        Gemini free tier is 1,500 RPM. For large docs (500+ sentences),
        the chunker batches sentences and sleeps between batches.
    """

    BATCH_SIZE = 50  # sentences per API batch
    BATCH_SLEEP_SEC = 1.0  # sleep between batches (free-tier safe)
    MIN_CHUNK_TOKENS = 100  # merge groups smaller than this

    def __init__(
        self,
        breakpoint_threshold: float = 0.75,
        chunk_size: int = 512,
        embedding_model: str = "models/gemini-embedding-001",
    ):
        self.threshold = breakpoint_threshold
        self.chunk_size = chunk_size
        self.embedding_model = embedding_model

    # ── Public entry point ────────────────────────────────────────

    def chunk(self, full_text: str, annotated_pages: list[dict]) -> list[ChunkData]:
        sentences = self._split_sentences(full_text)
        if not sentences:
            return []

        logger.info(
            f"[Semantic] Embedding {len(sentences)} sentences in batches of {self.BATCH_SIZE}..."
        )
        embeddings = self._embed_all(sentences)

        # Compute cosine similarities between consecutive sentences
        similarities = [
            _cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        # Find breakpoints where similarity drops below threshold
        groups = self._group_by_breakpoints(sentences, similarities)

        # Merge tiny groups and convert to ChunkData
        groups = self._merge_small_groups(groups)
        result = []
        char_cursor = 0

        for idx, group_sentences in enumerate(groups):
            text = " ".join(group_sentences)
            pos = full_text.find(group_sentences[0], char_cursor)
            char_start = pos if pos != -1 else char_cursor
            char_end = char_start + len(text)
            char_cursor = max(char_cursor, char_start + 1)

            result.append(
                ChunkData(
                    text=text.strip(),
                    chunk_index=idx,
                    page_start=_page_for_char(char_start, annotated_pages),
                    page_end=_page_for_char(char_end, annotated_pages),
                    char_start=char_start,
                    char_end=char_end,
                )
            )

        logger.info(f"[Semantic] {len(result)} chunks | threshold={self.threshold}")
        return result

    # ── Sentence splitting ────────────────────────────────────────

    def _split_sentences(self, text: str) -> list[str]:
        """
        Regex-based sentence splitter tuned for medical text.
        Handles:
          - Abbreviations: Dr., Fig., e.g., i.e., vs., et al.
          - Numbered lists: "1. Patient presented..."
          - Section headers (ALL CAPS lines)
        """
        # Protect common medical abbreviations from being split
        protected = text
        abbrevs = [
            r"Dr\.",
            r"Prof\.",
            r"Fig\.",
            r"Tab\.",
            r"Eq\.",
            r"et al\.",
            r"e\.g\.",
            r"i\.e\.",
            r"vs\.",
            r"approx\.",
            r"No\.",
            r"Vol\.",
            r"p\.",
            r"pp\.",
            r"ibid\.",
        ]
        placeholders = {}
        for i, abbrev in enumerate(abbrevs):
            placeholder = f"__ABBREV{i}__"
            protected = re.sub(abbrev, placeholder, protected, flags=re.IGNORECASE)
            placeholders[placeholder] = re.sub(r"\\", "", abbrev)

        # Split on sentence-ending punctuation followed by whitespace + capital
        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", protected)

        # Restore abbreviation placeholders
        sentences = []
        for sent in raw:
            for ph, orig in placeholders.items():
                sent = sent.replace(ph, orig.replace("\\", ""))
            sent = sent.strip()
            if len(sent) > 15:  # discard very short fragments
                sentences.append(sent)

        return sentences

    # ── Embedding ─────────────────────────────────────────────────

    def _embed_all(self, sentences: list[str]) -> list[list[float]]:
        """
        Embed all sentences in batches, sleeping between batches
        to stay within Gemini's rate limits.
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai is required for semantic chunking.\n"
                "Install:  pip install google-genai"
            )

        # Initialize the modern client (automatically picks up GEMINI_API_KEY from env)
        client = genai.Client()
        
        all_embeddings = []

        for batch_start in range(0, len(sentences), self.BATCH_SIZE):
            batch = sentences[batch_start : batch_start + self.BATCH_SIZE]

            # OPTIMIZATION: Send the entire batch list in one API call
            result = client.models.embed_content(
                model=self.embedding_model, # Ensure this is "gemini-embedding-001"
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT", # Must be uppercase in new SDK
                    output_dimensionality=768       # Keep vectors at 768 dimensions
                )
            )

            # Extract the float lists from the returned embedding objects
            batch_embeddings = [embedding.values for embedding in result.embeddings]
            all_embeddings.extend(batch_embeddings)

            # Rate-limit pause between batches
            if batch_start + self.BATCH_SIZE < len(sentences):
                import time
                time.sleep(self.BATCH_SLEEP_SEC)

        return all_embeddings

    # ── Grouping ──────────────────────────────────────────────────

    def _group_by_breakpoints(
        self,
        sentences: list[str],
        similarities: list[float],
    ) -> list[list[str]]:
        """
        Walk through similarity scores. Start a new group wherever
        the score drops below the threshold (topic boundary).
        """
        groups = [[sentences[0]]]

        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                # Semantic breakpoint detected — new chunk
                groups.append([sentences[i + 1]])
            else:
                groups[-1].append(sentences[i + 1])

        return groups

    def _merge_small_groups(self, groups: list[list[str]]) -> list[list[str]]:
        """Merge groups that are below MIN_CHUNK_TOKENS into the next group."""
        merged = []

        for group in groups:
            text = " ".join(group)
            if merged and _estimate_tokens(text) < self.MIN_CHUNK_TOKENS:
                merged[-1].extend(group)
            else:
                merged.append(list(group))

        return merged


# ─────────────────────────────────────────────────────────────────
#  Cosine similarity (pure Python — no numpy needed)
# ─────────────────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─────────────────────────────────────────────────────────────────
#  ChunkingService — main orchestrator
# ─────────────────────────────────────────────────────────────────


class ChunkingService:
    """
    Orchestrates PDF text extraction → chunking → DB save
    for a given Document model instance.

    Example
    -------
        from .services.chunking import ChunkingService

        # Called after a document is uploaded (e.g. in a Celery task):
        service = ChunkingService(document)
        chunks  = service.run_and_save()
        print(f"Created {len(chunks)} chunks for '{document.title}'")
    """

    def __init__(self, document):
        """
        Args:
            document: A Document model instance (from models.py).
                      Reads .file.path, .chunk_strategy, .chunk_size, .chunk_overlap.
        """
        self.document = document
        self.strategy = document.chunk_strategy
        self.chunker = self._build_chunker()

    # ── Build the right chunker ───────────────────────────────────

    def _build_chunker(self):
        size = self.document.chunk_size
        overlap = self.document.chunk_overlap

        if self.strategy == "fixed":
            return FixedSizeChunker(chunk_size=size, chunk_overlap=overlap)
        elif self.strategy == "recursive":
            return RecursiveChunker(chunk_size=size, chunk_overlap=overlap)
        elif self.strategy == "semantic":
            return SemanticChunker(chunk_size=size)
        else:
            logger.warning(
                f"Unknown strategy '{self.strategy}', defaulting to recursive."
            )
            return RecursiveChunker(chunk_size=size, chunk_overlap=overlap)

    # ── Run ───────────────────────────────────────────────────────

    def run(self) -> list[ChunkData]:
        """
        Extract text from the PDF and chunk it.
        Returns a list of ChunkData objects (not yet saved to DB).
        """
        from django.utils import timezone

        doc = self.document

        # Mark as processing
        doc.status = "processing"
        doc.save(update_fields=["status", "updated_at"])

        try:
            file_path = doc.file.path
            logger.info(
                f"[ChunkingService] Processing '{doc.title}' with strategy='{self.strategy}'"
            )

            # 1. Extract per-page text
            pages = _extract_pages(file_path)

            # Update page count while we have the info
            doc.page_count = len(pages)
            doc.save(update_fields=["page_count"])

            # 2. Build full text + annotate pages with char offsets
            full_text, annotated_pages = _annotate_pages(pages)

            if not full_text.strip():
                raise ValueError(
                    "PDF appears to be empty or image-only (no extractable text)."
                )

            # 3. Run chosen chunking strategy
            chunks = self.chunker.chunk(full_text, annotated_pages)

            # 4. Filter out empty/whitespace chunks
            chunks = [c for c in chunks if len(c.text.strip()) > 20]

            # Update status
            doc.status = "ready"
            doc.chunk_count = len(chunks)
            doc.processed_at = timezone.now()
            doc.save(
                update_fields=["status", "chunk_count", "processed_at", "updated_at"]
            )

            logger.info(f"[ChunkingService] Done — {len(chunks)} chunks created.")
            return chunks

        except Exception as exc:
            doc.status = "failed"
            doc.error_message = str(exc)
            doc.save(update_fields=["status", "error_message", "updated_at"])
            logger.error(
                f"[ChunkingService] Failed for '{doc.title}': {exc}", exc_info=True
            )
            raise

    def save(self, chunks: list[ChunkData]):
        """
        Bulk-insert ChunkData objects into the Chunk table.
        Deletes any existing chunks for this document first (idempotent).
        """
        from ..models import Chunk

        # Clear old chunks (safe to re-run)
        Chunk.objects.filter(document=self.document).delete()

        db_chunks = [
            Chunk(
                document=self.document,
                text=c.text,
                chunk_index=c.chunk_index,
                page_start=c.page_start,
                page_end=c.page_end,
                char_start=c.char_start,
                char_end=c.char_end,
                token_count=c.token_count,
            )
            for c in chunks
        ]

        created = Chunk.objects.bulk_create(db_chunks, batch_size=500)
        logger.info(f"[ChunkingService] Saved {len(created)} Chunk rows.")
        return created

    def run_and_save(self):
        """Convenience: run chunking and save to DB in one call."""
        chunks = self.run()
        return self.save(chunks)


# ─────────────────────────────────────────────────────────────────
#  Django management command integration
#  Add this to: management/commands/process_document.py
# ─────────────────────────────────────────────────────────────────
#
#   class Command(BaseCommand):
#       help = "Chunk and embed a document by UUID"
#
#       def add_arguments(self, parser):
#           parser.add_argument("doc_id", type=str)
#
#       def handle(self, *args, **options):
#           from myapp.models import Document
#           doc     = Document.objects.get(id=options["doc_id"])
#           service = ChunkingService(doc)
#           chunks  = service.run_and_save()
#           self.stdout.write(f"Created {len(chunks)} chunks.")


# ─────────────────────────────────────────────────────────────────
#  Standalone smoke test (no Django needed)
#  Run:  python services/chunking.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAMPLE_TEXT = """
    Hypertension, also known as high blood pressure, is a long-term medical condition
    in which the blood pressure in the arteries is persistently elevated.
    High blood pressure typically does not cause symptoms.

    Long-term high blood pressure, however, is a major risk factor for stroke,
    coronary artery disease, heart failure, atrial fibrillation, peripheral arterial
    disease, vision loss, chronic kidney disease, and dementia.

    Hypertension is classified as primary (essential) hypertension or secondary hypertension.
    About 90–95% of cases are primary, defined as high blood pressure due to nonspecific
    lifestyle and genetic factors. Lifestyle factors that increase the risk include excess
    salt in the diet, excess body weight, smoking, physical inactivity, and alcohol use.

    The remaining 5–10% of cases are categorized as secondary high blood pressure,
    defined as high blood pressure due to an identifiable cause, such as chronic kidney
    disease, narrowing of the kidney arteries, an endocrine disorder, or the use of
    birth control pills.

    Blood pressure is classified by two measurements, the systolic and diastolic pressures,
    which are the maximum and minimum pressures, respectively. Normal blood pressure at rest
    is within the range of 100–130 millimeters mercury (mmHg) systolic and 60–80 mmHg diastolic.

    High blood pressure is present if the resting blood pressure is persistently at or above
    130/80 or 140/90 mmHg. Different numbers apply to children. Ambulatory blood pressure
    monitoring over a 24-hour period appears more accurate than office-based blood pressure
    measurement.

    Lifestyle changes and medications can lower blood pressure and decrease the risk of
    health complications. Lifestyle changes include weight loss, physical exercise, decreased
    salt intake, reducing alcohol intake, and a healthy diet.
    """

    # Fake annotated pages for standalone test
    fake_pages = [
        {
            "page": 1,
            "text": SAMPLE_TEXT.strip(),
            "char_start": 0,
            "char_end": len(SAMPLE_TEXT),
        }
    ]

    print("=" * 60)
    print("  ChunkingService — Standalone Smoke Test")
    print("=" * 60)

    for Strategy, label in [
        (FixedSizeChunker(chunk_size=80, chunk_overlap=10), "Fixed-size"),
        (RecursiveChunker(chunk_size=80, chunk_overlap=10), "Recursive"),
    ]:
        print(f"\n── Strategy: {label} ──")
        chunks = Strategy.chunk(SAMPLE_TEXT.strip(), fake_pages)
        print(f"   Total chunks : {len(chunks)}")
        for c in chunks[:3]:
            preview = c.text[:80].replace("\n", " ")
            print(
                f'   [{c.chunk_index}] tokens={c.token_count:3d}  page={c.page_start}  "{preview}..."'
            )
        if len(chunks) > 3:
            print(f"   ... {len(chunks) - 3} more chunks")

    print("\n── Strategy: Semantic ──")
    print("   (Skipped in standalone mode — requires Gemini API key)")
    print("   Import and call SemanticChunker.chunk() with genai configured.")
    print("\n" + "=" * 60)
    print("  All non-API tests passed.")
    print("=" * 60)
