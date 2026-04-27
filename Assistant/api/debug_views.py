"""
api/debug_views.py
──────────────────────────────────────────────────────────────────
Debug API views — used only by the chunk inspector UI.
These endpoints expose internal chunking data for development.

Endpoints:
    GET  /api/debug/documents/              — list docs with chunk stats
    GET  /api/debug/documents/<id>/chunks/  — all chunks for a document
    GET  /api/debug/chunks/<id>/            — single chunk detail
    POST /api/debug/documents/<id>/rechunk/ — re-run pipeline with new settings
"""

import json

from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator


@method_decorator(csrf_exempt, name="dispatch")
class DebugDocumentListView(View):
    """
    GET /api/debug/documents/
    Lists all documents with chunking statistics for the debug UI.
    """

    def get(self, request):
        from ..models import Document

        docs = Document.objects.prefetch_related("chunks").order_by("-created_at")

        data = []
        for doc in docs:
            chunks     = doc.chunks.all()
            token_list = list(chunks.values_list("token_count", flat=True))

            data.append({
                "id":             str(doc.id),
                "title":          doc.title,
                "status":         doc.status,
                "chunk_strategy": doc.chunk_strategy,
                "chunk_size":     doc.chunk_size,
                "chunk_overlap":  doc.chunk_overlap,
                "page_count":     doc.page_count,
                "chunk_count":    doc.chunk_count,
                "file_size_kb":   doc.file_size_kb,
                "created_at":     doc.created_at.isoformat(),
                "processed_at":   doc.processed_at.isoformat() if doc.processed_at else None,
                "error_message":  doc.error_message,
                "token_stats":    _token_stats(token_list),
            })

        return JsonResponse({"documents": data})


@method_decorator(csrf_exempt, name="dispatch")
class DebugChunkListView(View):
    """
    GET /api/debug/documents/<doc_id>/chunks/
    Returns all chunks for a document with full metadata.
    Supports ?page_filter=3 to show only chunks from a specific page.
    """

    def get(self, request, doc_id):
        from ..models import Document, Chunk

        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            return JsonResponse({"error": "Document not found"}, status=404)

        qs = doc.chunks.order_by("chunk_index")

        # Optional page filter
        page_filter = request.GET.get("page_filter")
        if page_filter:
            try:
                pf = int(page_filter)
                qs = qs.filter(page_start__lte=pf, page_end__gte=pf)
            except ValueError:
                pass

        chunks = []
        for c in qs:
            chunks.append({
                "id":          str(c.id),
                "chunk_index": c.chunk_index,
                "text":        c.text,
                "token_count": c.token_count,
                "page_start":  c.page_start,
                "page_end":    c.page_end,
                "char_start":  c.char_start,
                "char_end":    c.char_end,
                "chroma_id":   c.chroma_id,
                "is_embedded": c.is_embedded,
                "preview":     c.text[:120] + ("..." if len(c.text) > 120 else ""),
            })

        token_list = [c["token_count"] for c in chunks]

        return JsonResponse({
            "document": {
                "id":             str(doc.id),
                "title":          doc.title,
                "status":         doc.status,
                "chunk_strategy": doc.chunk_strategy,
                "chunk_size":     doc.chunk_size,
                "chunk_overlap":  doc.chunk_overlap,
                "page_count":     doc.page_count,
                "error_message":  doc.error_message,
            },
            "total_chunks": len(chunks),
            "token_stats":  _token_stats(token_list),
            "chunks":       chunks,
        })


@method_decorator(csrf_exempt, name="dispatch")
class DebugRechunkView(View):
    """
    POST /api/debug/documents/<doc_id>/rechunk/
    Re-runs the chunking pipeline with updated settings.

    Body (JSON):
        {
            "chunk_strategy": "recursive",
            "chunk_size":     512,
            "chunk_overlap":  64
        }
    """

    def post(self, request, doc_id):
        from ..models import Document
        from ..services.chunk_pipeline import run_pipeline

        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            return JsonResponse({"error": "Document not found"}, status=404)

        if doc.status == "processing":
            return JsonResponse({"error": "Document is already processing"}, status=409)

        # Parse settings override
        try:
            body = json.loads(request.body or "{}")
        except json.JSONDecodeError:
            body = {}

        if "chunk_strategy" in body:
            doc.chunk_strategy = body["chunk_strategy"]
        if "chunk_size" in body:
            doc.chunk_size = int(body["chunk_size"])
        if "chunk_overlap" in body:
            doc.chunk_overlap = int(body["chunk_overlap"])

        # Reset status so pipeline runs again
        doc.status = "pending"
        doc.save()

        try:
            summary = run_pipeline(doc)
            return JsonResponse({
                "success": True,
                "summary": summary,
                "chunk_count": doc.chunk_count,
            })
        except Exception as exc:
            return JsonResponse({
                "success": False,
                "error": str(exc),
            }, status=500)


# ── Helper ─────────────────────────────────────────────────────────

def _token_stats(token_list: list[int]) -> dict:
    """Compute min / max / avg / median token counts for a set of chunks."""
    if not token_list:
        return {"min": 0, "max": 0, "avg": 0, "median": 0, "total": 0}

    sorted_t = sorted(token_list)
    n        = len(sorted_t)
    median   = sorted_t[n // 2] if n % 2 else (sorted_t[n // 2 - 1] + sorted_t[n // 2]) // 2

    return {
        "min":    min(token_list),
        "max":    max(token_list),
        "avg":    round(sum(token_list) / n),
        "median": median,
        "total":  sum(token_list),
    }