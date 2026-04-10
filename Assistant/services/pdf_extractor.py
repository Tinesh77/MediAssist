"""
services/pdf_extractor.py
──────────────────────────────────────────────────────────────────
PDFExtractor — extracts clean, structured text from medical PDFs.

Handles:
  - Multi-page extraction with per-page char offsets
  - Table detection (preserves tabular structure as pipe-delimited text)
  - Header / footer removal heuristic (repeated short lines)
  - Image-only page detection (warns, skips gracefully)
  - Encoding normalisation (ligatures, smart quotes, etc.)

Usage:
    from .services.pdf_extractor import PDFExtractor

    extractor = PDFExtractor("path/to/document.pdf")
    result    = extractor.extract()

    result.full_text          # entire document as one string
    result.pages              # list of PageData objects
    result.page_count         # int
    result.char_count         # int
    result.has_text           # False if PDF is image-only

Standalone test:
    python services/pdf_extractor.py path/to/file.pdf
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────

@dataclass
class PageData:
    """All extracted data for one PDF page."""
    page_number:  int
    text:         str                        # cleaned prose text
    tables:       list[str] = field(default_factory=list)  # table strings
    char_start:   int = 0                   # offset in full_text
    char_end:     int = 0
    word_count:   int = 0
    is_empty:     bool = False              # True if no text found (image page)

    def __post_init__(self):
        self.word_count = len(self.text.split()) if self.text else 0


@dataclass
class ExtractionResult:
    """Complete extraction result for one document."""
    pages:       list[PageData]
    full_text:   str
    page_count:  int
    char_count:  int
    has_text:    bool
    empty_pages: list[int] = field(default_factory=list)  # page nums with no text
    warnings:    list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────
#  Text cleaning helpers
# ─────────────────────────────────────────────────────────────────

# Unicode ligatures and common OCR artefacts found in medical PDFs
_LIGATURE_MAP = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
    "\u00ad": "",   # soft hyphen
    "\u00a0": " ",  # non-breaking space
})


def _clean_text(text: str) -> str:
    """
    Normalise raw PDF text:
      1. Fix ligatures and smart quotes
      2. Remove excessive whitespace while keeping paragraph breaks
      3. Remove null bytes and control characters
      4. Strip hyphenated line-breaks (re-join split words)
      
    """
    if not text:
        return ""

    # Fix ligatures
    text = text.translate(_LIGATURE_MAP)

    # Re-join words broken across lines with a hyphen
    text = re.sub(r"-\n(\s*)", "", text)

    # Collapse runs of spaces (but keep newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ newlines → double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove control characters except newline and tab
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)

    return text.strip()


def _table_to_text(table: list[list]) -> str:
    """
    Convert a pdfplumber table (list of rows, each row a list of cells)
    into a pipe-delimited string that's easy for the LLM to read.

    Example output:
        | Drug | Dosage | Frequency |
        | Metformin | 500 mg | Twice daily |
    """
    if not table:
        return ""

    rows = []
    for row in table:
        cells = [str(cell).strip() if cell is not None else "" for cell in row]
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


def _is_header_or_footer(line: str, page_height: float, y_pos: float) -> bool:
    """
    Heuristic: lines in the top 8% or bottom 8% of the page
    that are very short are likely headers / footers / page numbers.
    """
    if page_height <= 0:
        return False
    relative_pos = y_pos / page_height
    is_edge = relative_pos < 0.08 or relative_pos > 0.92
    is_short = len(line.strip()) < 60
    return is_edge and is_short


# ─────────────────────────────────────────────────────────────────
#  PDFExtractor
# ─────────────────────────────────────────────────────────────────

class PDFExtractor:
    """
    Extracts structured text from a PDF file using pdfplumber.

    Args:
        file_path:           Path to the PDF (str or Path).
        remove_headers:      Attempt to strip repeated header/footer lines.
        extract_tables:      Include table content as pipe-delimited text.
        min_chars_per_page:  Pages with fewer chars are flagged as empty.
    """

    def __init__(
        self,
        file_path:          str | Path,
        remove_headers:     bool = True,
        extract_tables:     bool = True,
        min_chars_per_page: int  = 30,
    ):
        self.file_path          = Path(file_path)
        self.remove_headers     = remove_headers
        self.extract_tables     = extract_tables
        self.min_chars_per_page = min_chars_per_page

        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.file_path}")

    # ── Main entry point ──────────────────────────────────────────

    def extract(self) -> ExtractionResult:
        """
        Run full extraction. Returns an ExtractionResult.
        Never raises — errors per page are caught and logged.
        
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required.\n"
                "Install:  pip install pdfplumber"
            )

        pages_data:  list[PageData] = []
        empty_pages: list[int]      = []
        warnings:    list[str]      = []
        char_offset: int            = 0

        logger.info(f"[PDFExtractor] Opening: {self.file_path.name}")

        with pdfplumber.open(self.file_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"[PDFExtractor] {total_pages} pages detected")

            # Collect all page texts first — needed for header dedup
            raw_page_texts = []
            for page in pdf.pages:
                try:
                    raw_page_texts.append(page.extract_text() or "")
                except Exception:
                    raw_page_texts.append("")

            # Detect repeated lines (headers/footers) across pages
            repeated_lines: set[str] = set()
            if self.remove_headers and total_pages > 2:
                repeated_lines = self._find_repeated_lines(raw_page_texts)
                if repeated_lines:
                    logger.info(
                        f"[PDFExtractor] Removing {len(repeated_lines)} repeated header/footer lines"
                    )

            # ── Per-page extraction ───────────────────────────────
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    page_data = self._extract_page(
                        page        = page,
                        page_number = page_num,
                        char_offset = char_offset,
                        repeated    = repeated_lines,
                    )
                except Exception as exc:
                    logger.warning(f"[PDFExtractor] Page {page_num} failed: {exc}")
                    warnings.append(f"Page {page_num}: extraction error — {exc}")
                    page_data = PageData(
                        page_number = page_num,
                        text        = "",
                        char_start  = char_offset,
                        char_end    = char_offset,
                        is_empty    = True,
                    )

                pages_data.append(page_data)

                if page_data.is_empty:
                    empty_pages.append(page_num)

                # Advance char offset (+1 for the \n separator between pages)
                char_offset = page_data.char_end + 1

        # Build full text from all pages
        full_text = "\n".join(p.text for p in pages_data if p.text)
        has_text  = bool(full_text.strip())

        if not has_text:
            warnings.append(
                "No text could be extracted. This PDF may be image-only (scanned). "
                "Consider running OCR (e.g. pytesseract) before chunking."
            )

        if empty_pages:
            warnings.append(
                f"Pages with no extractable text: {empty_pages}. "
                f"These may be images, charts, or blank pages."
            )

        logger.info(
            f"[PDFExtractor] Complete — "
            f"{len(pages_data)} pages, "
            f"{len(full_text):,} chars, "
            f"{len(empty_pages)} empty pages"
        )

        return ExtractionResult(
            pages       = pages_data,
            full_text   = full_text,
            page_count  = len(pages_data),
            char_count  = len(full_text),
            has_text    = has_text,
            empty_pages = empty_pages,
            warnings    = warnings,
        )

    # ── Per-page logic ────────────────────────────────────────────

    def _extract_page(
        self,
        page:        object,
        page_number: int,
        char_offset: int,
        repeated:    set[str],
    ) -> PageData:
        """Extract and clean text for a single pdfplumber page object."""

        # 1. Extract tables first (before raw text, so we can exclude table boxes)
        table_texts: list[str] = []
        table_bboxes: list     = []

        if self.extract_tables:
            tables = page.extract_tables() or []
            for table in tables:
                t_text = _table_to_text(table)
                if t_text.strip():
                    table_texts.append(t_text)

            # Get bounding boxes so we can exclude them from prose extraction
            try:
                for t_finder in (page.find_tables() or []):
                    table_bboxes.append(t_finder.bbox)
            except Exception:
                pass

        # 2. Extract prose text (exclude table regions if possible)
        if table_bboxes:
            try:
                # Crop away table areas and extract remaining text
                remaining = page
                for bbox in table_bboxes:
                    remaining = remaining.outside_bbox(bbox)
                raw_text = remaining.extract_text() or ""
            except Exception:
                raw_text = page.extract_text() or ""
        else:
            raw_text = page.extract_text() or ""

        # 3. Remove repeated header/footer lines
        if repeated and raw_text:
            lines    = raw_text.split("\n")
            filtered = [ln for ln in lines if ln.strip() not in repeated]
            raw_text = "\n".join(filtered)

        # 4. Clean text
        prose_text = _clean_text(raw_text)

        # 5. Combine prose + tables
        parts = [prose_text] + table_texts
        full_page_text = "\n\n".join(p for p in parts if p.strip())

        # 6. Detect empty pages
        is_empty = len(full_page_text.strip()) < self.min_chars_per_page

        char_start = char_offset
        char_end   = char_start + len(full_page_text)

        return PageData(
            page_number = page_number,
            text        = full_page_text,
            tables      = table_texts,
            char_start  = char_start,
            char_end    = char_end,
            is_empty    = is_empty,
        )

    # ── Header / footer deduplication ────────────────────────────

    def _find_repeated_lines(self, page_texts: list[str]) -> set[str]:
        """
        Find lines that appear on more than 40% of pages —
        almost certainly headers, footers, or page numbers.
    
        """
        from collections import Counter

        threshold   = max(2, len(page_texts) * 0.4)
        line_counts = Counter()

        for text in page_texts:
            for line in set(text.split("\n")):
                stripped = line.strip()
                if stripped and len(stripped) < 80:
                    line_counts[stripped] += 1

        return {line for line, count in line_counts.items() if count >= threshold}


# ─────────────────────────────────────────────────────────────────
#  Django integration helper
# ─────────────────────────────────────────────────────────────────

def extract_from_document(document) -> ExtractionResult:
    """
    Convenience wrapper: pass a Document model instance,
    get back an ExtractionResult.

    Updates document.page_count and document.warnings on the model.

    Usage:
        from .services.pdf_extractor import extract_from_document
        result = extract_from_document(document)
    """
    extractor = PDFExtractor(document.file.path)
    result    = extractor.extract()

    # Persist page count
    document.page_count = result.page_count
    document.save(update_fields=["page_count"])

    if result.warnings:
        logger.warning(
            f"[extract_from_document] Warnings for '{document.title}': "
            + " | ".join(result.warnings)
        )

    return result


# ─────────────────────────────────────────────────────────────────
#  Standalone smoke test
#  Run:  python services/pdf_extractor.py path/to/file.pdf
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <path_to_pdf>")
        sys.exit(1)

    path      = sys.argv[1]
    extractor = PDFExtractor(path)
    result    = extractor.extract()

    print("=" * 60)
    print(f"  File        : {path}")
    print(f"  Pages       : {result.page_count}")
    print(f"  Characters  : {result.char_count:,}")
    print(f"  Has text    : {result.has_text}")
    print(f"  Empty pages : {result.empty_pages or 'None'}")
    print("=" * 60)

    for page in result.pages[:3]:
        print(f"\n── Page {page.page_number} ──")
        print(f"   Words      : {page.word_count}")
        print(f"   Char range : [{page.char_start} – {page.char_end}]")
        print(f"   Tables     : {len(page.tables)}")
        preview = page.text[:300].replace("\n", " ")
        print(f"   Preview    : {preview}...")

    if len(result.pages) > 3:
        print(f"\n... {len(result.pages) - 3} more pages")

    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  ! {w}")

    print("\n" + "=" * 60)
    print("  Extraction complete.")
    print("=" * 60)
    