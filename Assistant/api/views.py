from django.utils import timezone
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView

from ..models import Document
from .serializers import DocumentListSerializer, DocumentUploadSerializer


class DocumentUploadView(APIView):
    """
    POST /api/documents/upload/

    Accepts a multipart form with a PDF file.
    Validates type, size, and magic bytes, then creates
    a Document record with status=PENDING.

    The actual chunking + embedding happens in a background
    task (Celery or Django management command) — not here.

    Request (multipart/form-data):
        file            required  — the PDF file
        chunk_strategy  optional  — fixed | recursive | semantic  (default: recursive)
        chunk_size      optional  — token size per chunk           (default: 512)
        chunk_overlap   optional  — overlap between chunks         (default: 64)

    Response 201:
        {
            "success": true,
            "message": "Document uploaded successfully. Processing will begin shortly.",
            "document": { ...DocumentUploadSerializer fields... }
        }
    """

    parser_classes = [MultiPartParser, FormParser]

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

        # TODO Phase 2+: trigger async chunking task here
        # from ..tasks import process_document
        # process_document.delay(str(doc.id))

        return Response(
            {
                "success": True,
                "message": "Document uploaded successfully. Processing will begin shortly.",
                "document": DocumentUploadSerializer(doc).data,
            },
            status=status.HTTP_201_CREATED,
        )


class DocumentListView(APIView):
    """
    GET /api/documents/

    Returns all documents for the current user (or all if anonymous).
    Ordered by most recently uploaded.
    """

    def get(self, request):
        qs = Document.objects.all()

        if request.user.is_authenticated:
            qs = qs.filter(uploaded_by=request.user)

        serializer = DocumentListSerializer(qs, many=True)

        return Response(
            {
                "success": True,
                "count":   qs.count(),
                "results": serializer.data,
            }
        )


class DocumentDetailView(APIView):
    """
    GET    /api/documents/<doc_id>/   — retrieve document + chunk summary
    DELETE /api/documents/<doc_id>/   — soft-delete (sets is_active=False)
    """

    def _get_doc(self, doc_id, request):
        try:
            qs = Document.objects.all()
            if request.user.is_authenticated:
                qs = qs.filter(uploaded_by=request.user)
            return qs.get(id=doc_id)
        except Document.DoesNotExist:
            return None

    def get(self, request, doc_id):
        doc = self._get_doc(doc_id, request)
        if not doc:
            return Response(
                {"success": False, "error": "Document not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        data = DocumentListSerializer(doc).data

        # Include a small chunk preview for the debug UI
        chunks_preview = list(
            doc.chunks.order_by("chunk_index").values(
                "chunk_index", "page_start", "page_end", "token_count", "is_embedded"
            )[:10]
        )
        data["chunks_preview"] = chunks_preview

        return Response({"success": True, "document": data})

    def delete(self, request, doc_id):
        doc = self._get_doc(doc_id, request)
        if not doc:
            return Response(
                {"success": False, "error": "Document not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        doc.delete()
        return Response(
            {"success": True, "message": f"Document '{doc.title}' deleted."},
            status=status.HTTP_200_OK,
        )
