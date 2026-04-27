"""
api/retrieval_views.py
──────────────────────────────────────────────────────────────────
REST API endpoints for Phase 3 — indexing and semantic search.

Endpoints:
    POST /api/documents/<id>/index/     — embed + index a document
    POST /api/search/                   — semantic similarity search
    GET  /api/search/stats/             — ChromaDB collection stats
"""

import json

from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator


@method_decorator(csrf_exempt, name="dispatch")
class IndexDocumentView(View):
    """
    POST /api/documents/<doc_id>/index/

    Triggers embedding + ChromaDB indexing for a document.
    The document must already be chunked (status=ready).

    Response 200:
        {
            "success": true,
            "chunks_indexed": 47,
            "collection_total": 312,
            "message": "..."
        }
    """

    def post(self, request, doc_id):
        from ..models import Document
        from ..services.retrieval_service import ChromaService, IndexingService

        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            return JsonResponse({"success": False, "error": "Document not found."}, status=404)

        if doc.status != "ready":
            return JsonResponse({
                "success": False,
                "error": (
                    f"Document status is '{doc.status}'. "
                    f"Must be 'ready' (chunked) before indexing. "
                    f"Run the chunking pipeline first."
                )
            }, status=400)

        try:
            count = IndexingService.index_document(doc)
            return JsonResponse({
                "success":          True,
                "chunks_indexed":   count,
                "collection_total": ChromaService.collection_count(),
                "message": (
                    f"Successfully indexed {count} chunks for '{doc.title}'. "
                    f"Document is ready for semantic search."
                ),
            })
        except Exception as exc:
            return JsonResponse({
                "success": False,
                "error":   str(exc),
            }, status=500)


@method_decorator(csrf_exempt, name="dispatch")
class SemanticSearchView(View):
    """
    POST /api/search/

    Run semantic similarity search against indexed documents.

    Request body (JSON):
        {
            "query":        "What are the symptoms of hypertension?",  required
            "document_id":  "uuid-string",      optional — scope to one doc
            "document_ids": ["uuid1", "uuid2"], optional — scope to multiple docs
            "top_k":        20                  optional — default 20
        }

    Response 200:
        {
            "success": true,
            "query": "...",
            "total_results": 5,
            "results": [
                {
                    "rank":        1,
                    "score":       0.912,
                    "chunk_index": 14,
                    "text":        "...",
                    "preview":     "...",
                    "page_start":  3,
                    "page_end":    3,
                    "token_count": 284,
                    "chroma_id":   "chunk_...",
                    "document": {
                        "id":    "uuid",
                        "title": "Hypertension Guidelines 2024"
                    }
                },
                ...
            ]
        }
    """

    def post(self, request):
        from ..services.retrieval_service import RetrievalService , _get_strategy

        # ── Parse request ─────────────────────────────────────────
        try:
            body = json.loads(request.body or "{}")
        except json.JSONDecodeError:
            return JsonResponse(
                {"success": False, "error": "Invalid JSON body."},
                status=400
            )

        query = body.get("query", "").strip()
        if not query:
            return JsonResponse(
                {"success": False, "error": "'query' field is required."},
                status=400
            )

        top_k        = min(int(body.get("top_k", 20)), 50)
        document_id  = body.get("document_id")
        document_ids = body.get("document_ids", [])
        user_id      = request.user.id if request.user.is_authenticated else None

        # ── Resolve document scope ────────────────────────────────
        document = None
        if document_id:
            from ..models import Document
            try:
                document = Document.objects.get(id=document_id)
            except Document.DoesNotExist:
                return JsonResponse(
                    {"success": False, "error": f"Document '{document_id}' not found."},
                    status=404
                )

        # ── Run search ────────────────────────────────────────────
        try:
            results = RetrievalService.search(
                query        = query,
                top_k        = top_k,
                document     = document,
                document_ids = document_ids or None,
                user_id      = user_id,
            )
        except ValueError as exc:
            return JsonResponse(
                {"success": False, "error": str(exc)},
                status=400
            )
        except Exception as exc:
            return JsonResponse(
                {"success": False, "error": f"Search failed: {exc}"},
                status=500
            )

        # ── Serialise results — full source info for Phase 4 ──────
        serialised = []
        for rank, r in enumerate(results, start=1):
            page_start = r.chunk.page_start
            page_end   = r.chunk.page_end

            serialised.append({
                # Rank + score
                "rank":         rank,
                "score":        r.score,

                # Chunk content
                "text":         r.chunk.text,
                "preview":      r.preview,
                "chunk_index":  r.chunk.chunk_index,
                "token_count":  r.chunk.token_count,

                # Source page — what the task required
                "page_start":   page_start,
                "page_end":     page_end,
                "page_label":   f"p.{page_start}" if page_start == page_end
                                else f"p.{page_start}–{page_end}",

                # Character offsets (for highlight-in-PDF in Phase 5)
                "char_start":   r.chunk.char_start,
                "char_end":     r.chunk.char_end,

                # Source citation — used by Phase 4 PromptBuilder
                "source_label": r.source_label,
                "chroma_id":    r.chroma_id,

                # Document info
                "document": {
                    "id":    str(r.chunk.document.id),
                    "title": r.chunk.document.title,
                },
            })

        return JsonResponse({
            "success":       True,
            "query":         query,
            "total_results": len(serialised),
            "collection_strategy": _get_strategy(),
            "results":       serialised,
        })


@method_decorator(csrf_exempt, name="dispatch")
class SearchStatsView(View):
    """
    GET /api/search/stats/
    Returns ChromaDB collection statistics — useful for the debug UI.
    """

    def get(self, request):
        from ..models import Document
        from ..services.retrieval_service import ChromaService

        try:
            total_vectors = ChromaService.collection_count()

            # Per-document breakdown
            doc_stats = []
            for doc in Document.objects.filter(status="ready").order_by("-created_at"):
                vec_count      = ChromaService.document_vector_count(str(doc.id))
                embedded_count = doc.chunks.filter(is_embedded=True).count()
                doc_stats.append({
                    "id":              str(doc.id),
                    "title":           doc.title,
                    "chunk_count":     doc.chunk_count,
                    "embedded_chunks": embedded_count,
                    "vector_count":    vec_count,
                    "fully_indexed":   vec_count == doc.chunk_count,
                })

            return JsonResponse({
                "success":              True,
                "total_vectors":        total_vectors,
                "chroma_collection":    "mediassist",
                "embedding_model":      "models/embedding-001",
                "embedding_dimensions": 768,
                "documents":            doc_stats,
            })

        except Exception as exc:
            return JsonResponse(
                {"success": False, "error": str(exc)},
                status=500
            )