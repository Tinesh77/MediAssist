from django.utils import timezone
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny
from rest_framework.authentication import BasicAuthentication
from rest_framework.response import Response
from rest_framework.views import APIView
from django.shortcuts import render
import threading
import logging

logger = logging.getLogger(__name__)

from ..models import Document
from .serializers import DocumentListSerializer, DocumentUploadSerializer


# ── Background worker ─────────────────────────────────────────────

def index(request):
    return render(request, "index.html")

def chunk_inspector(request):
    return render(request, "chunk_inspector.html")



def _run_pipeline_and_index(doc_id: str, user_id: int):
    """
    Runs in a background thread after upload:
      1. ChunkPipeline  → extract text + save Chunk rows
      2. IndexingService → embed chunks + store vectors in ChromaDB

    Using a thread keeps the upload response fast (instant 201)
    while processing happens in the background.

    For production, replace this thread with a Celery task:
        from .tasks import process_and_index
        process_and_index.delay(doc_id, user_id)
    """
    try:
        import django
        django.setup()
    except Exception:
        pass   # already set up in the main thread

    try:
        from ..models import Document
        from ..services.chunk_pipeline import run_pipeline
        from ..services.retrieval_service import IndexingService

        doc = Document.objects.get(id=doc_id)

        # Step 1 — chunk the PDF
        logger.info(f"[Upload Worker] Starting chunking for '{doc.title}'")
        run_pipeline(doc)

        # Step 2 — embed + index into ChromaDB
        logger.info(f"[Upload Worker] Starting indexing for '{doc.title}'")
        IndexingService.index_document(doc, user_id=user_id)

        logger.info(f"[Upload Worker] Complete for '{doc.title}'")

    except Exception as exc:
        logger.error(f"[Upload Worker] Failed for doc {doc_id}: {exc}", exc_info=True)
        try:
            doc = Document.objects.get(id=doc_id)
            doc.status        = "failed"
            doc.error_message = str(exc)
            doc.save(update_fields=["status", "error_message"])
        except Exception:
            pass


class DocumentUploadView(APIView):
    """
    POST /api/documents/upload/

    Accepts a multipart form with a PDF file.
    Validates type, size, and magic bytes, then creates
    a Document record with status=PENDING.

    Authentication: AllowAny — no login required for local dev.
    The authentication_classes = [] override removes DRF's default
    SessionAuthentication, which enforces CSRF on every POST from
    a browser. Without this, every upload returns 403 Forbidden
    even when CORS is configured correctly.
    """

    # ── Gap 1 fix: explicit AllowAny so unauthenticated POSTs work ──
    permission_classes     = [AllowAny]
    # ── Gap 2 fix: empty authentication bypasses CSRF enforcement ───
    authentication_classes = []
    parser_classes         = [MultiPartParser, FormParser]

    def post(self, request):
        serializer = DocumentUploadSerializer(
            data=request.data,
            context={"request": request},
        )

        if not serializer.is_valid():
            return Response(
                {
                    "success": False,
                    "errors":  serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        doc = serializer.save(
            uploaded_by=request.user if request.user.is_authenticated else None,
        )

        # ── Auto-trigger chunk + index pipeline in background ─────
        # The upload returns 201 immediately.
        # Chunking + embedding runs in a daemon thread so the user
        # gets instant feedback and can poll document.status to track progress.
        user_id = request.user.id if request.user.is_authenticated else None

        worker = threading.Thread(
            target  = _run_pipeline_and_index,
            args    = (str(doc.id), user_id),
            daemon  = True,
            name    = f"pipeline-{doc.id}",
        )
        worker.start()
        logger.info(f"[UploadView] Started background pipeline thread for '{doc.title}'")

        return Response(
            {
                "success": True,
                "message": (
                    "Document uploaded. Chunking and embedding are running in "
                    "the background. Poll GET /api/documents/<id>/ and check "
                    "'status' — it will change from 'processing' → 'ready'."
                ),
                "document": DocumentUploadSerializer(doc).data,
            },
            status=status.HTTP_201_CREATED,
        )


class DocumentListView(APIView):
    """
    GET /api/documents/
    Returns all documents. For anonymous dev use, returns everything.
    """
    permission_classes     = [AllowAny]
    authentication_classes = []

    def get(self, request):
        # Return all documents in dev. In production with auth enabled,
        # filter by uploaded_by=request.user instead.
        qs         = Document.objects.all().order_by("-created_at")
        serializer = DocumentListSerializer(qs, many=True)
        return Response({"success": True, "count": qs.count(), "results": serializer.data})


class DocumentDetailView(APIView):
    """
    GET    /api/documents/<doc_id>/   — retrieve document + chunk summary
    DELETE /api/documents/<doc_id>/   — delete document
    """
    permission_classes     = [AllowAny]
    authentication_classes = []

    def _get_doc(self, doc_id):
        try:
            return Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            return None

    def get(self, request, doc_id):
        doc = self._get_doc(doc_id)
        if not doc:
            return Response({"success": False, "error": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        data = DocumentListSerializer(doc).data
        chunks_preview = list(
            doc.chunks.order_by("chunk_index").values(
                "chunk_index", "page_start", "page_end", "token_count", "is_embedded"
            )[:10]
        )
        data["chunks_preview"] = chunks_preview
        return Response({"success": True, "document": data})

    def delete(self, request, doc_id):
        doc = self._get_doc(doc_id)
        if not doc:
            return Response({"success": False, "error": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        doc.delete()
        return Response({"success": True, "message": f"Document '{doc.title}' deleted."}, status=status.HTTP_200_OK)