"""
services/prompt_builder.py
──────────────────────────────────────────────────────────────────
Phase 4 of the MediAssist RAG pipeline.

Contains three classes:

    ReRanker         — takes top-20 candidates from ChromaDB,
                       re-scores them by true relevance,
                       returns the best top-k for the prompt.

    PromptBuilder    — assembles the final prompt that Gemini receives:
                       system role + CoT instruction + context chunks
                       + citation rules + conversation history + question.

    ChatService      — orchestrates the full RAG turn:
                       retrieve → re-rank → build prompt → call Gemini
                       → parse citations → persist Message rows.

Full pipeline for one user turn:
    User question
        → RetrievalService.search()      top-20 candidates from Chroma
        → ReRanker.rerank()              narrows to top-5 by relevance
        → PromptBuilder.build()          assembles the Gemini prompt
        → ChatService.answer()           calls Gemini, streams or returns
        → Message.save()                 persists turn + source chunks

Usage (in your Django view):
    from .services.prompt_builder import ChatService

    response = ChatService.answer(
        question = "What are the first-line treatments for hypertension?",
        session  = chat_session,          # ChatSession model instance
        document = document,              # scope retrieval to one doc
    )

    response.answer          # str — Gemini's full answer
    response.sources         # list of SourceChunk dicts (text, page, title, score)
    response.citations_used  # list of [Source N] labels found in the answer
    response.tokens_used     # {"prompt": int, "completion": int, "total": int}

Standalone smoke test:
    python services/prompt_builder.py
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Generator, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────

CHAT_MODEL         = "gemini-2.5-flash"  # Modern fast reasoning model
MAX_CONTEXT_TOKENS = 6000    # hard cap on context injected into prompt
TOP_K_RETRIEVE     = 20      # how many to fetch from ChromaDB
TOP_K_RERANK       = 5       # how many to pass to Gemini after re-ranking
HISTORY_TURNS      = 6       # last N conversation turns included

# sentence-transformers cross-encoder model
# fast (22 MB), no GPU needed, runs on CPU in ~50 ms per batch
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ─────────────────────────────────────────────────────────────────
#  SourceChunk — the citation unit
# ─────────────────────────────────────────────────────────────────

@dataclass
class SourceChunk:
    """
    One retrieved + re-ranked chunk ready to inject into the prompt.
    """
    n:               int
    text:            str
    page_start:      int
    page_end:        int
    doc_title:       str
    doc_id:          str
    chunk_index:     int
    chroma_id:       str        # Gap 1 fix: was missing, caused AttributeError crash
    retrieval_score: float
    rerank_score:    float = 0.0

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p.{self.page_start}"
        return f"p.{self.page_start}-{self.page_end}"

    @property
    def citation_label(self) -> str:
        return f"[Source {self.n}]"

    @property
    def header(self) -> str:
        return (
            f"[Source {self.n}] {self.doc_title} - {self.page_label} "
            f"(relevance: {self.rerank_score:.2f})"
        )

    @property
    def source_label(self) -> str:
        """Human-readable citation label. Used by frontend chip display."""
        return f"{self.doc_title} - {self.page_label}"

    def to_dict(self) -> dict:
        """Serialisable dict for API responses and Message.retrieval_scores."""
        return {
            "n":               self.n,
            "citation_label":  self.citation_label,
            "source_label":    self.source_label,    # Gap 2 fix: was missing
            "text":            self.text,
            "page_start":      self.page_start,
            "page_end":        self.page_end,
            "page_label":      self.page_label,
            "doc_title":       self.doc_title,
            "doc_id":          self.doc_id,
            "chunk_index":     self.chunk_index,
            "chroma_id":       self.chroma_id,       # Gap 2 fix: was missing
            "retrieval_score": round(self.retrieval_score, 4),
            "rerank_score":    round(self.rerank_score, 4),
        }


# ─────────────────────────────────────────────────────────────────
#  ChatResponse — return value from ChatService.answer()
# ─────────────────────────────────────────────────────────────────

@dataclass
class ChatResponse:
    """
    The complete result of one RAG turn.
    """
    answer:           str
    sources:          list[dict] = field(default_factory=list)
    citations_used:   list[str]  = field(default_factory=list)
    tokens_used:      dict       = field(default_factory=dict)
    retrieval_scores: list[dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────
#  ReRanker
# ─────────────────────────────────────────────────────────────────

class ReRanker:
    """
    Second-pass relevance scoring on top of ChromaDB cosine similarity.
    """

    _cross_encoder = None
    _ce_available  = None   # None=unchecked | True=ready | False=unavailable

    @classmethod
    def _get_cross_encoder(cls):
        """
        Lazy-load CrossEncoder. Returns model or None.
        Result cached — model only loaded once per process.
        """
        if cls._ce_available is True:
            return cls._cross_encoder
        if cls._ce_available is False:
            return None

        try:
            from sentence_transformers import CrossEncoder as CE
            logger.info(f"[ReRanker] Loading '{CROSS_ENCODER_MODEL}' ...")
            cls._cross_encoder = CE(CROSS_ENCODER_MODEL, max_length=512)
            cls._ce_available  = True
            logger.info("[ReRanker] CrossEncoder ready")
            return cls._cross_encoder
        except ImportError:
            logger.warning(
                "[ReRanker] sentence-transformers not installed → "
                "falling back to Gemini. "
                "Fix: pip install sentence-transformers"
            )
            cls._ce_available = False
            return None
        except Exception as exc:
            logger.warning(f"[ReRanker] CrossEncoder load failed ({exc}) → Gemini fallback")
            cls._ce_available = False
            return None

    @classmethod
    def rerank(
        cls,
        question:   str,
        candidates: list,
        top_k:      int = TOP_K_RERANK,
    ) -> list["SourceChunk"]:
        """
        Re-rank candidates, return top_k as SourceChunk objects.
        Citation numbers (n=1,2,3...) are assigned here in rank order.
        """
        if not candidates:
            return []

        top_k  = min(top_k, len(candidates))
        scored = None
        strategy = "unknown"

        # ── A: sentence-transformers CrossEncoder ─────────────────
        ce = cls._get_cross_encoder()
        if ce is not None:
            try:
                scored   = cls._cross_encoder_score(question, candidates, ce)
                strategy = "sentence-transformers/cross-encoder"
            except Exception as exc:
                logger.warning(f"[ReRanker] CrossEncoder predict failed: {exc}")

        # ── B: Gemini listwise ────────────────────────────────────
        if scored is None:
            try:
                scored   = cls._gemini_score(question, candidates)
                strategy = f"{CHAT_MODEL}/listwise"
            except Exception as exc:
                logger.warning(f"[ReRanker] Gemini scoring failed: {exc}")

        # ── C: weighted fallback ──────────────────────────────────
        if scored is None:
            scored   = cls._fallback_score(candidates)
            strategy = "weighted-fallback"

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        result = []
        for n, (candidate, rerank_score) in enumerate(top, start=1):
            chunk = candidate.chunk
            result.append(SourceChunk(
                n               = n,
                text            = chunk.text,
                page_start      = chunk.page_start,
                page_end        = chunk.page_end,
                doc_title       = chunk.document.title,
                doc_id          = str(chunk.document.id),
                chunk_index     = chunk.chunk_index,
                chroma_id       = chunk.chroma_id,
                retrieval_score = candidate.score,
                rerank_score    = round(rerank_score, 4),
            ))

        logger.info(
            f"[ReRanker] {len(candidates)} → {len(result)} | "
            f"strategy={strategy} | "
            f"top={result[0].rerank_score:.4f}"
        )
        return result

    # ── Strategy A: sentence-transformers CrossEncoder ────────────

    @classmethod
    def _cross_encoder_score(
        cls,
        question:     str,
        candidates:   list,
        cross_encoder,
    ) -> list[tuple]:
        """Score (question, passage) pairs with a CrossEncoder."""
        pairs = [
            (question, c.chunk.text[:512])   # hard-truncate to model max_length
            for c in candidates
        ]

        raw_scores = cross_encoder.predict(pairs)

        min_s = float(min(raw_scores))
        max_s = float(max(raw_scores))
        span  = (max_s - min_s) if max_s != min_s else 1.0
        norm  = [round((float(s) - min_s) / span, 4) for s in raw_scores]

        return list(zip(candidates, norm))

    # ── Strategy B: Gemini listwise scoring ──────────────────────

    @classmethod
    def _gemini_score(cls, question: str, candidates: list) -> list[tuple]:
        """
        ONE Gemini call scores all candidates simultaneously.
        """
        from google.genai import types
        from MediAssist.gemini import GeminiService
        GeminiService._ensure_init()

        lines = [
            "You are a relevance scorer for a medical information retrieval system.",
            "Score each passage 0–10 for how directly it answers the question.",
            "10 = directly and completely answers it. 0 = completely irrelevant.",
            "Respond with ONLY a comma-separated list of numbers — nothing else.",
            "",
            f"QUESTION: {question}",
            "",
        ]
        for i, c in enumerate(candidates, start=1):
            preview = c.chunk.text[:300].replace("\n", " ")
            lines.append(f"PASSAGE {i}: {preview}")

        lines.append("\nSCORES:")

        client = GeminiService._client
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents="\n".join(lines)
        )
        raw = response.text.strip()

        nums  = re.findall(r"\d+(?:\.\d+)?", raw)
        scores = [min(10.0, max(0.0, float(s))) for s in nums]

        while len(scores) < len(candidates):
            scores.append(0.0)
        scores = scores[:len(candidates)]

        norm = [round(s / 10.0, 4) for s in scores]
        return list(zip(candidates, norm))

    # ── Strategy C: weighted fallback ────────────────────────────

    @classmethod
    def _fallback_score(cls, candidates: list) -> list[tuple]:
        """No API call. Blends cosine similarity + position bonus."""
        n = len(candidates)
        return [
            (c, round(0.85 * c.score + 0.15 * (n - i) / n, 4))
            for i, c in enumerate(candidates)
        ]


# ─────────────────────────────────────────────────────────────────
#  PromptBuilder
# ─────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Assembles the complete prompt sent to Gemini for each RAG turn.
    """

    SYSTEM_PROMPT = """You are MediAssist, an expert AI healthcare assistant.

Your role is to answer medical questions accurately and safely using ONLY the information provided in the context passages below. You have deep expertise in medicine, pharmacology, and clinical practice.

SAFETY RULES (non-negotiable):
- Base every factual claim strictly on the provided context. Never invent information.
- If the context does not contain the answer, say clearly: "The provided documents do not contain information about this."
- For clinical decisions (dosing, diagnosis, treatment), always add: "Please consult a qualified healthcare professional before acting on this information."
- Do not provide specific dosage recommendations without citing the exact source passage.

CITATION RULES (mandatory):
- After every factual claim, immediately cite the source: [Source 1], [Source 2], etc.
- You may cite multiple sources for one claim: [Source 1][Source 3]
- Never make a factual claim without a citation.
- Citations must match the [Source N] numbers in the context — do not invent source numbers.

OUTPUT FORMAT:
- Write in clear, professional medical English.
- Use short paragraphs. Use bullet points for lists of symptoms, treatments, or criteria.
- End your response with a "## Sources used" section listing which sources you cited and why."""

    COT_INSTRUCTIONS = """Before writing your answer, work through these steps (you may show your reasoning):

STEP 1 — IDENTIFY: Which of the provided sources directly address the question? List their numbers.
STEP 2 — SYNTHESISE: What do these sources collectively say about the question? Note any contradictions.
STEP 3 — ANSWER: Write a clear, cited answer using only what the sources support.

If a step finds no relevant information, state that explicitly rather than inferring."""

    @classmethod
    def build(
        cls,
        question:   str,
        chunks:     list[SourceChunk],
        history:    list[dict] = None,
        extra_instructions: str = "",
    ) -> tuple[str, str]:
        """
        Build the system prompt and user message for a Gemini API call.
        """
        system = cls.SYSTEM_PROMPT
        if extra_instructions:
            system += f"\n\nADDITIONAL INSTRUCTIONS:\n{extra_instructions}"

        context_block = cls._build_context_block(chunks)
        history_block = cls._build_history_block(history or [])

        user_message = "\n\n".join(filter(None, [
            "## CONTEXT PASSAGES",
            context_block,
            "## INSTRUCTIONS",
            cls.COT_INSTRUCTIONS,
            history_block,
            f"## QUESTION\n{question}",
            "## YOUR ANSWER",
        ]))

        token_estimate = len(user_message) // 4
        logger.info(
            f"[PromptBuilder] Built prompt — "
            f"{len(chunks)} chunks | "
            f"~{token_estimate} tokens | "
            f"{len(history or [])} history turns"
        )
        return system, user_message

    @classmethod
    def _build_context_block(cls, chunks: list[SourceChunk]) -> str:
        """Inject chunks into the prompt with [Source N] headers."""
        if not chunks:
            return "No relevant context passages found."

        separator  = "─" * 50
        blocks     = []
        used_tokens = 0

        for sc in chunks:
            chunk_tokens = len(sc.text) // 4 + 20

            if used_tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
                remaining_chars = (MAX_CONTEXT_TOKENS - used_tokens) * 4 - 80
                if remaining_chars < 200:
                    logger.info(
                        f"[PromptBuilder] Token budget reached at chunk {sc.n} — "
                        f"skipping remaining chunks"
                    )
                    break
                truncated_text = sc.text[:remaining_chars] + "\n[... truncated for length ...]"
                block = f"{separator}\n{sc.header}\n{separator}\n{truncated_text}"
                blocks.append(block)
                used_tokens += remaining_chars // 4
                break

            block = f"{separator}\n{sc.header}\n{separator}\n{sc.text}"
            blocks.append(block)
            used_tokens += chunk_tokens

        logger.info(
            f"[PromptBuilder] Context block: {len(blocks)}/{len(chunks)} chunks, "
            f"~{used_tokens} tokens"
        )
        return "\n\n".join(blocks)

    @classmethod
    def _build_history_block(cls, history: list[dict]) -> str:
        """Format conversation history as a readable block."""
        if not history:
            return ""

        recent = history[-(HISTORY_TURNS * 2):]
        lines  = ["## CONVERSATION HISTORY"]

        for turn in recent:
            role    = turn.get("role", "user")
            text    = turn.get("parts", [{}])[0].get("text", "")
            label   = "User" if role == "user" else "Assistant"
            preview = text[:400] + ("..." if len(text) > 400 else "")
            lines.append(f"{label}: {preview}")

        return "\n".join(lines)

    @classmethod
    def extract_citations(cls, answer_text: str) -> list[str]:
        """Extract all [Source N] citations from Gemini's answer."""
        found  = re.findall(r"\[Source\s+\d+\]", answer_text)
        seen   = set()
        unique = []
        for c in found:
            normalised = re.sub(r"\s+", " ", c).strip()
            if normalised not in seen:
                seen.add(normalised)
                unique.append(normalised)
        return unique


# ─────────────────────────────────────────────────────────────────
#  ChatService
# ─────────────────────────────────────────────────────────────────

class ChatService:
    """
    Orchestrates a complete RAG question-answering turn.
    """

    @classmethod
    def answer(
        cls,
        question:    str,
        session,
        document     = None,
        document_ids: list[str] = None,
        user_id:     int = None,
        top_k_retrieve: int = TOP_K_RETRIEVE,
        top_k_rerank:   int = TOP_K_RERANK,
        extra_instructions: str = "",
    ) -> ChatResponse:
        """
        Full RAG turn — retrieve, re-rank, build prompt, call Gemini.
        """
        from .retrieval_service import RetrievalService
        from google.genai import types
        from MediAssist.gemini import GeminiService

        GeminiService._ensure_init()

        # ── Step 1: Retrieve candidates ───────────────────────────
        logger.info(
            f"[ChatService] Question: '{question[:80]}' | "
            f"session={session.id}"
        )

        candidates = RetrievalService.search(
            query        = question,
            top_k        = top_k_retrieve,
            document     = document,
            document_ids = document_ids,
            user_id      = user_id,
        )

        if not candidates:
            logger.warning("[ChatService] No candidates retrieved — answering without context")
            no_context_answer = (
                "I could not find relevant information in the uploaded documents "
                "to answer your question. Please ensure the relevant documents have "
                "been uploaded and indexed, or try rephrasing your question."
            )
            cls._save_messages(session, question, no_context_answer, [], [], {})
            return ChatResponse(
                answer         = no_context_answer,
                sources        = [],
                citations_used = [],
                tokens_used    = {},
            )

        # ── Step 2: Re-rank to top_k_rerank ──────────────────────
        ranked_chunks = ReRanker.rerank(
            question   = question,
            candidates = candidates,
            top_k      = top_k_rerank,
        )

        # ── Step 3: Build prompt ──────────────────────────────────
        history = session.get_history_for_prompt(last_n=HISTORY_TURNS)

        system_prompt, user_message = PromptBuilder.build(
            question            = question,
            chunks              = ranked_chunks,
            history             = history,
            extra_instructions  = extra_instructions,
        )

        # ── Step 4: Call Gemini ───────────────────────────────────
        logger.info(f"[ChatService] Calling Gemini ({CHAT_MODEL})...")

        client = GeminiService._client
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
        )
        answer = response.text

        # Extract token usage (Gemini returns usage_metadata)
        tokens_used = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            tokens_used = {
                "prompt":     getattr(um, "prompt_token_count",     0),
                "completion": getattr(um, "candidates_token_count", 0),
                "total":      getattr(um, "total_token_count",      0),
            }
            logger.info(
                f"[ChatService] Tokens — "
                f"prompt={tokens_used['prompt']} "
                f"completion={tokens_used['completion']} "
                f"total={tokens_used['total']}"
            )

        # ── Step 5: Extract citations + map to specific chunks ────
        citations_used = PromptBuilder.extract_citations(answer)
        logger.info(f"[ChatService] Citations found: {citations_used}")

        citation_map = {sc.citation_label: sc for sc in ranked_chunks}
        cited_chunks = [
            citation_map[label]
            for label in citations_used
            if label in citation_map
        ]
        uncited = [sc for sc in ranked_chunks if sc.citation_label not in citations_used]

        logger.info(
            f"[ChatService] {len(cited_chunks)}/{len(ranked_chunks)} "
            f"chunks cited | uncited: {[sc.n for sc in uncited]}"
        )

        # ── Step 6: Build full retrieval_scores record ────────────
        ranked_by_chroma = {sc.chroma_id: sc for sc in ranked_chunks}
        retrieval_scores = []
        for c in candidates:
            ranked = ranked_by_chroma.get(c.chunk.chroma_id)
            retrieval_scores.append({
                "chroma_id":       c.chunk.chroma_id,
                "chunk_index":     c.chunk.chunk_index,
                "page_start":      c.chunk.page_start,
                "doc_title":       c.chunk.document.title,
                "retrieval_score": round(c.score, 4),
                "rerank_score":    ranked.rerank_score if ranked else 0.0,
                "citation_n":      ranked.n if ranked else None,
                "was_cited":       ranked.citation_label in citations_used if ranked else False,
            })

        # ── Step 7: Persist messages ──────────────────────────────
        source_dicts = [sc.to_dict() for sc in ranked_chunks]

        cls._save_messages(
            session          = session,
            question         = question,
            answer           = answer,
            ranked_chunks    = ranked_chunks,
            cited_chunks     = cited_chunks,
            tokens_used      = tokens_used,
            retrieval_scores = retrieval_scores,
        )

        if not session.title and session.messages.count() <= 2:
            session.title = question[:80]
            session.save(update_fields=["title"])

        return ChatResponse(
            answer           = answer,
            sources          = source_dicts,
            citations_used   = citations_used,
            tokens_used      = tokens_used,
            retrieval_scores = retrieval_scores,
        )

    # ── Streaming version ─────────────────────────────────────────

    @classmethod
    def stream(
        cls,
        question:    str,
        session,
        document     = None,
        document_ids: list[str] = None,
        user_id:     int = None,
        top_k_retrieve: int = TOP_K_RETRIEVE,
        top_k_rerank:   int = TOP_K_RERANK,
    ) -> Generator[str, None, ChatResponse]:
        """
        Streaming RAG turn — yields text tokens as Gemini generates them.
        """
        from .retrieval_service import RetrievalService
        from google.genai import types
        from MediAssist.gemini import GeminiService

        GeminiService._ensure_init()

        candidates = RetrievalService.search(
            query        = question,
            top_k        = top_k_retrieve,
            document     = document,
            document_ids = document_ids,
            user_id      = user_id,
        )

        if not candidates:
            no_context = (
                "I could not find relevant information in the uploaded documents. "
                "Please ensure relevant documents are uploaded and indexed."
            )
            yield no_context
            cls._save_messages(session, question, no_context, [], [], {})
            return ChatResponse(answer=no_context, sources=[], citations_used=[], tokens_used={})

        ranked_chunks = ReRanker.rerank(
            question   = question,
            candidates = candidates,
            top_k      = top_k_rerank,
        )

        history = session.get_history_for_prompt(last_n=HISTORY_TURNS)
        system_prompt, user_message = PromptBuilder.build(
            question = question,
            chunks   = ranked_chunks,
            history  = history,
        )

        client = GeminiService._client
        stream_response = client.models.generate_content_stream(
            model=CHAT_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
        )

        full_answer = ""
        for chunk in stream_response:
            if chunk.text:
                full_answer += chunk.text
                yield chunk.text

        citations_used  = PromptBuilder.extract_citations(full_answer)
        citation_map    = {sc.citation_label: sc for sc in ranked_chunks}
        cited_chunks    = [
            citation_map[label]
            for label in citations_used
            if label in citation_map
        ]
        source_dicts = [sc.to_dict() for sc in ranked_chunks]

        ranked_by_chroma = {sc.chroma_id: sc for sc in ranked_chunks}
        retrieval_scores = [
            {
                "chroma_id":       c.chunk.chroma_id,
                "chunk_index":     c.chunk.chunk_index,
                "page_start":      c.chunk.page_start,
                "doc_title":       c.chunk.document.title,
                "retrieval_score": round(c.score, 4),
                "rerank_score":    ranked_by_chroma[c.chunk.chroma_id].rerank_score
                                   if c.chunk.chroma_id in ranked_by_chroma else 0.0,
                "citation_n":      ranked_by_chroma[c.chunk.chroma_id].n
                                   if c.chunk.chroma_id in ranked_by_chroma else None,
                "was_cited":       citation_map.get(
                                       ranked_by_chroma[c.chunk.chroma_id].citation_label
                                   ) is not None
                                   if c.chunk.chroma_id in ranked_by_chroma else False,
            }
            for c in candidates
        ]

        cls._save_messages(
            session          = session,
            question         = question,
            answer           = full_answer,
            ranked_chunks    = ranked_chunks,
            cited_chunks     = cited_chunks,
            tokens_used      = {},
            retrieval_scores = retrieval_scores,
        )

        if not session.title:
            session.title = question[:80]
            session.save(update_fields=["title"])

        return ChatResponse(
            answer           = full_answer,
            sources          = source_dicts,
            citations_used   = citations_used,
            tokens_used      = {},
            retrieval_scores = retrieval_scores,
        )

    @classmethod
    def _save_messages(
        cls,
        session,
        question:        str,
        answer:          str,
        ranked_chunks:   list,
        cited_chunks:    list,
        tokens_used:     dict,
        retrieval_scores: list[dict] = None,
        was_streamed:    bool = False,
    ):
        """Persist user + assistant Message rows in PostgreSQL."""
        from django.utils import timezone
        from ..models import Message, Chunk

        Message.objects.create(
            session = session,
            role    = Message.Role.USER,
            content = question,
        )

        assistant_msg = Message.objects.create(
            session           = session,
            role              = Message.Role.ASSISTANT,
            content           = answer,
            prompt_tokens     = tokens_used.get("prompt",     0),
            completion_tokens = tokens_used.get("completion", 0),
            total_tokens      = tokens_used.get("total",      0),
            retrieval_scores  = retrieval_scores or [],
            was_streamed      = was_streamed,
        )

        if cited_chunks:
            cited_chroma_ids = [sc.chroma_id for sc in cited_chunks]
            cited_chunk_objs = Chunk.objects.filter(chroma_id__in=cited_chroma_ids)
            assistant_msg.source_chunks.set(cited_chunk_objs)

            logger.info(
                f"[ChatService] Linked {cited_chunk_objs.count()} "
                f"cited chunks to message {assistant_msg.id} "
                f"| citation labels: {[sc.citation_label for sc in cited_chunks]}"
            )
        else:
            logger.info(
                f"[ChatService] No citations found in answer — "
                f"source_chunks M2M left empty for message {assistant_msg.id}"
            )

        session.updated_at = timezone.now()
        session.save(update_fields=["updated_at"])


# ─────────────────────────────────────────────────────────────────
#  Standalone smoke test
#  Run: python services/prompt_builder.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 65)
    print("  Phase 4 — PromptBuilder Smoke Test")
    print("=" * 65)

    fake_chunks = [
        SourceChunk(
            n=1, text=(
                "Hypertension is defined as a sustained systolic blood pressure "
                "≥130 mmHg or diastolic ≥80 mmHg. First-line pharmacological "
                "treatment includes ACE inhibitors, ARBs, calcium channel blockers, "
                "and thiazide diuretics. Lifestyle modifications such as DASH diet, "
                "exercise, and sodium restriction are recommended alongside medication."
            ),
            page_start=3, page_end=3,
            doc_title="JNC 8 Hypertension Guidelines",
            doc_id="abc-123", chunk_index=14, chroma_id="test-1",
            retrieval_score=0.91, rerank_score=0.93,
        ),
        SourceChunk(
            n=2, text=(
                "ACE inhibitors (e.g., lisinopril, enalapril) are recommended as "
                "first-line therapy in patients with diabetes or chronic kidney disease. "
                "Common side effects include dry cough (10–15% of patients) and "
                "hyperkalaemia. Contraindicated in pregnancy and bilateral renal artery stenosis."
            ),
            page_start=7, page_end=7,
            doc_title="Drug Reference Manual 2024",
            doc_id="def-456", chunk_index=22, chroma_id="test-2",
            retrieval_score=0.87, rerank_score=0.88,
        ),
        SourceChunk(
            n=3, text=(
                "The SPRINT trial demonstrated that intensive blood pressure control "
                "(target SBP <120 mmHg) significantly reduced cardiovascular events "
                "compared to standard control (<140 mmHg), but was associated with "
                "higher rates of acute kidney injury and electrolyte abnormalities."
            ),
            page_start=12, page_end=13,
            doc_title="Cardiovascular Outcomes Research Review",
            doc_id="ghi-789", chunk_index=8, chroma_id="test-3",
            retrieval_score=0.83, rerank_score=0.80,
        ),
    ]

    question = "What are the first-line treatments for hypertension in diabetic patients?"

    print("\n[1] PromptBuilder.build() ...")
    system, user_msg = PromptBuilder.build(
        question = question,
        chunks   = fake_chunks,
        history  = [
            {"role": "user",  "parts": [{"text": "What is hypertension?"}]},
            {"role": "model", "parts": [{"text": "Hypertension is elevated blood pressure above 130/80 mmHg [Source 1]."}]},
        ],
    )
    print(f"   [OK] System prompt:  {len(system):,} chars")
    print(f"   [OK] User message:   {len(user_msg):,} chars")
    print(f"   [OK] Token estimate: ~{len(user_msg) // 4:,} tokens")

    print("\n[2] PromptBuilder.extract_citations() ...")
    sample_answer = (
        "ACE inhibitors are first-line for diabetic patients [Source 1][Source 2]. "
        "The SPRINT trial supports intensive control [Source 3]. "
        "Always consult a physician [Source 1]."
    )
    citations = PromptBuilder.extract_citations(sample_answer)
    print(f"   [OK] Found: {citations}")
    assert citations == ["[Source 1]", "[Source 2]", "[Source 3]"], "Citation extraction failed!"

    print("\n[3] ReRanker._fallback_score() ...")

    class FakeSearchResult:
        def __init__(self, score, text):
            self.score = score
            self.chunk = type("Chunk", (), {
                "text": text, "page_start": 1, "page_end": 1,
                "document": type("Doc", (), {"title": "Test", "id": "x"})(),
                "chunk_index": 0, "chroma_id": "test-c"
            })()

    fake_candidates = [
        FakeSearchResult(0.91, "High blood pressure treatment with ACE inhibitors."),
        FakeSearchResult(0.85, "Diabetes management includes metformin."),
        FakeSearchResult(0.78, "Hypertension lifestyle changes: DASH diet, exercise."),
    ]
    fallback = ReRanker._fallback_score(fake_candidates)
    fallback.sort(key=lambda x: x[1], reverse=True)
    print(f"   [OK] Scored {len(fallback)} candidates")

    print("\n[4] Full Gemini prompt test ...")
    try:
        from google import genai
        from google.genai import types
        import os

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=system,
            )
        )
        answer   = response.text

        citations = PromptBuilder.extract_citations(answer)
        print(f"   [OK] Gemini responded ({len(answer)} chars)")
        print(f"   [OK] Citations in answer: {citations}")

    except Exception as e:
        print(f"   [FAIL] Gemini call failed: {e}")

    print("\n" + "=" * 65)
    print("  Smoke test complete.")
    print("=" * 65)