"""
services/chunk_pipeline.py
──────────────────────────────────────────────────────────────────
ChunkPipeline — end-to-end pipeline that:
  1. Extracts text from a Document's PDF via PDFExtractor
  2. Chunks the text using ChunkingService (fixed / recursive / semantic)
  3. Saves every chunk to the Chunk model with full page + position metadata

This is the single function you call after a document is uploaded.

Usage:
    from .services.chunk_pipeline import run_pipeline

    # In a Celery task or management command:
    summary = run_pipeline(document)
    print(summary)
    # → "Created 47 chunks for 'Hypertension Guidelines.pdf' (recursive, 12 pages)"

Django management command:
    python manage.py chunk_document <document_uuid>
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
#  run_pipeline  — main entry point
# ─────────────────────────────────────────────────────────────────

def run_pipeline(document) -> str:
    """
    Full extract → chunk → save pipeline for one Document instance.

    Args:
        document:  A Document model instance (status must not be 'processing').

    Returns:
        A human-readable summary string (useful for management commands / logs).

    Raises:
        Any exception from extraction or chunking — callers should catch
        and handle (the pipeline sets document.status = 'failed' before raising).
    """
    from django.utils import timezone

    from .pdf_extractor import PDFExtractor, OCREngine
    from .chunking import (
        ChunkingService,
        FixedSizeChunker,
        RecursiveChunker,
        SemanticChunker,
        ChunkData,
    )
    from ..models import Chunk, Document

    doc = document

    # ── Guard: don't re-process if already ready ─────────────────
    if doc.status == "ready":
        logger.info(f"[Pipeline] '{doc.title}' already processed — skipping.")
        return f"Skipped: '{doc.title}' already has {doc.chunk_count} chunks."

    # ── Mark as processing ────────────────────────────────────────
    doc.status = "processing"
    doc.save(update_fields=["status", "updated_at"])

    try:
        start_time = datetime.now()

        # ── Step 1: PDF extraction ────────────────────────────────
        logger.info(f"[Pipeline] Step 1/3 — extracting text from '{doc.title}'")

        extractor = PDFExtractor(
            file_path      = doc.file.path,
            remove_headers = True,
            extract_tables = True,
        )
        result = extractor.extract()

        # ── OCR retry: if pdfplumber got nothing, try OCR ─────────
        # This handles scanned/image-only PDFs (lab reports, prescriptions).
        if not result.has_text:
            if OCREngine.available():
                logger.info(
                    f"[Pipeline] pdfplumber found no text — retrying with OCR "
                    f"(tesseract) for '{doc.title}'"
                )
                extractor_ocr = PDFExtractor(
                    file_path      = doc.file.path,
                    remove_headers = False,   # OCR output doesn't need header removal
                    extract_tables = False,   # pdfplumber tables N/A for image pages
                    force_ocr      = True,    # skip pdfplumber, go straight to tesseract
                )
                result = extractor_ocr.extract()
            else:
                logger.error(
                    f"[Pipeline] No text in '{doc.title}' and OCR is not available. "
                    f"Install: pip install pymupdf pytesseract pillow + tesseract binary."
                )

        # ── Final check after OCR attempt ─────────────────────────
        if not result.has_text:
            if OCREngine.available():
                raise ValueError(
                    "OCR ran but could not extract any text. "
                    "The PDF may be too low quality or corrupted. "
                    "Try scanning at higher resolution (300 DPI or above)."
                )
            else:
                raise ValueError(
                    "This PDF appears to be image-based (scanned) and OCR is not installed. "
                    "Fix: pip install pymupdf pytesseract pillow, then install the tesseract "
                    "binary from https://github.com/UB-Mannheim/tesseract/wiki (Windows) "
                    "or: sudo apt install tesseract-ocr (Linux). "
                    "Also add to settings.py: "
                    "pytesseract.pytesseract.tesseract_cmd = "
                    r"r'C:/Program Files/Tesseract-OCR/tesseract.exe'"
                )

        # Persist page count now that we have it
        doc.page_count = result.page_count
        doc.save(update_fields=["page_count"])

        logger.info(
            f"[Pipeline] Extracted {result.char_count:,} chars "
            f"from {result.page_count} pages"
            f"{' (OCR)' if result.ocr_used else ''}"
        )

        # ── Step 2: Build chunker ─────────────────────────────────
        logger.info(
            f"[Pipeline] Step 2/3 — chunking "
            f"(strategy={doc.chunk_strategy}, "
            f"size={doc.chunk_size}, "
            f"overlap={doc.chunk_overlap})"
        )

        chunker = _build_chunker(doc)
        raw_chunks: list[ChunkData] = chunker.chunk(
            full_text      = result.full_text,
            annotated_pages = _to_annotated_pages(result.pages),
        )

        # Filter trivially short chunks
        chunks = [c for c in raw_chunks if len(c.text.strip()) > 20]
        logger.info(f"[Pipeline] Produced {len(chunks)} chunks")

        # ── Step 3: Save to DB ────────────────────────────────────
        logger.info(f"[Pipeline] Step 3/3 — saving chunks to database")

        _save_chunks(doc, chunks)

        # ── Mark document as ready ────────────────────────────────
        elapsed = (datetime.now() - start_time).total_seconds()

        doc.status       = "ready"
        doc.chunk_count  = len(chunks)
        doc.processed_at = timezone.now()
        doc.error_message = ""
        doc.save(update_fields=["status", "chunk_count", "processed_at", "error_message", "updated_at"])

        summary = (
            f"Created {len(chunks)} chunks for '{doc.title}' "
            f"({doc.chunk_strategy}, {result.page_count} pages, {elapsed:.1f}s)"
        )
        logger.info(f"[Pipeline] Done — {summary}")
        return summary

    except Exception as exc:
        doc.status        = "failed"
        doc.error_message = str(exc)
        doc.save(update_fields=["status", "error_message", "updated_at"])
        logger.error(f"[Pipeline] Failed for '{doc.title}': {exc}", exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────
#  _save_chunks  — bulk insert with full metadata
# ─────────────────────────────────────────────────────────────────

def _save_chunks(document, chunks: list) -> list:
    """
    Delete any existing chunks for this document, then bulk-create
    new Chunk rows from the ChunkData list.

    Sets chroma_id to a stable, predictable string so the Phase 3
    embedding step can match DB rows to ChromaDB entries without
    a separate lookup.
    """
    from ..models import Chunk

    # Idempotent — safe to re-run
    deleted, _ = Chunk.objects.filter(document=document).delete()
    if deleted:
        logger.info(f"[Pipeline] Deleted {deleted} old chunks before re-processing")

    db_objects = []
    for c in chunks:
        db_objects.append(Chunk(
            document    = document,
            text        = c.text,
            chunk_index = c.chunk_index,
            page_start  = c.page_start,
            page_end    = c.page_end,
            char_start  = c.char_start,
            char_end    = c.char_end,
            token_count = c.token_count,
            # Stable ID used by ChromaDB in Phase 3
            chroma_id   = f"chunk_{document.id}_{c.chunk_index}",
        ))

    created = Chunk.objects.bulk_create(db_objects, batch_size=500)
    logger.info(f"[Pipeline] Saved {len(created)} Chunk rows (bulk_create)")
    return created


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def _build_chunker(document):
    """Instantiate the correct chunker from the document's config fields."""
    from .chunking import FixedSizeChunker, RecursiveChunker, SemanticChunker

    size    = document.chunk_size
    overlap = document.chunk_overlap

    if document.chunk_strategy == "fixed":
        return FixedSizeChunker(chunk_size=size, chunk_overlap=overlap)
    elif document.chunk_strategy == "semantic":
        return SemanticChunker(chunk_size=size)
    else:
        return RecursiveChunker(chunk_size=size, chunk_overlap=overlap)


def _to_annotated_pages(pages) -> list[dict]:
    """
    
    Convert PageData objects (from PDFExtractor) into the annotated-page
    dict format expected by the chunker classes.

    """
    return [
        {
            "page":       p.page_number,
            "text":       p.text,
            "char_start": p.char_start,
            "char_end":   p.char_end,
        }
        for p in pages
    ]