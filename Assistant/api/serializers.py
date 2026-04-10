import os
from rest_framework import serializers
from ..models import Document


MAX_FILE_SIZE_MB = 20
ALLOWED_TYPES    = {"application/pdf"}
ALLOWED_EXTS     = {".pdf"}


class DocumentUploadSerializer(serializers.ModelSerializer):
    """
    Validates and creates a Document on upload.
    Enforces: PDF only, max 20 MB, sane filename.
    """

    file = serializers.FileField()

    class Meta:
        model  = Document
        fields = [
            "id",
            "title",
            "file",
            "chunk_strategy",
            "chunk_size",
            "chunk_overlap",
        ]
        read_only_fields = ["id"]
        extra_kwargs = {
            "title":          {"required": False},
            "chunk_strategy": {"required": False},
            "chunk_size":     {"required": False},
            "chunk_overlap":  {"required": False},
        }

    # ── File validation ────────────────────────────────────────────

    def validate_file(self, file):
        # 1. Extension check
        ext = os.path.splitext(file.name)[1].lower()
        if ext not in ALLOWED_EXTS:
            raise serializers.ValidationError(
                f"Unsupported file type '{ext}'. Only PDF files are accepted."
            )

        # 2. MIME type check (read from content_type sent by browser)
        content_type = getattr(file, "content_type", "")
        if content_type and content_type not in ALLOWED_TYPES:
            raise serializers.ValidationError(
                f"Invalid MIME type '{content_type}'. Expected application/pdf."
            )

        # 3. Magic bytes check — first 4 bytes of a PDF are always %PDF
        header = file.read(4)
        file.seek(0)
        if header != b"%PDF":
            raise serializers.ValidationError(
                "File content does not appear to be a valid PDF."
            )

        # 4. Size check
        size_mb = file.size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise serializers.ValidationError(
                f"File is {size_mb:.1f} MB. Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
            )

        return file

    # ── Auto-fill title from filename ──────────────────────────────

    def validate(self, attrs):
        if not attrs.get("title"):
            raw_name   = attrs["file"].name
            attrs["title"] = os.path.splitext(raw_name)[0].replace("_", " ").replace("-", " ").title()
        return attrs

    # ── Save with extra computed fields ───────────────────────────

    def create(self, validated_data):
        file        = validated_data["file"]
        size_kb     = file.size // 1024

        doc = Document.objects.create(
            **validated_data,
            file_type    = "pdf",
            file_size_kb = size_kb,
        )
        return doc


class DocumentListSerializer(serializers.ModelSerializer):
    """Compact serializer for listing documents."""

    class Meta:
        model  = Document
        fields = [
            "id",
            "title",
            "file_size_kb",
            "page_count",
            "chunk_count",
            "status",
            "chunk_strategy",
            "created_at",
            "processed_at",
        ]