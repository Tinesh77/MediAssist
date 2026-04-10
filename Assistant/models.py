from django.db import models
import uuid
from django.contrib.auth.models import User

# Create your models here.


class Document(models.Model):

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    class ChunkStrategy(models.TextChoices):
        FIXED = "fixed", "Fixed-size"
        RECURSIVE = "recursive", "Recursive"
        SEMANTIC = "semantic", "Semantic"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="documents",
        null=True,
        blank=True,
    )

    # File info
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to="medical_docs/%Y/%m/%d/")
    file_type = models.CharField(max_length=10, default="pdf")  # pdf, txt, docx
    file_size_kb = models.PositiveIntegerField(default=0)
    page_count = models.PositiveIntegerField(default=0)

    # RAG config
    chunk_strategy = models.CharField(
        max_length=20,
        choices=ChunkStrategy.choices,
        default=ChunkStrategy.RECURSIVE,
    )
    chunk_size = models.PositiveIntegerField(default=512)  # tokens
    chunk_overlap = models.PositiveIntegerField(default=64)  # tokens

    # Processing state
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )
    error_message = models.TextField(blank=True, default="")
    chunk_count = models.PositiveIntegerField(default=0)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "rag_document"
        ordering = ["-created_at"]
        verbose_name = "Document"
        verbose_name_plural = "Documents"

    def __str__(self):
        return f"{self.title} [{self.status}]"

    @property
    def is_ready(self):
        return self.status == self.Status.READY

    @property
    def filename(self):
        if self.file:
            return self.file.name.split("/")[-1]
        return ""


class Chunk(models.Model):

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="chunks",
    )

    # Content
    text = models.TextField()
    token_count = models.PositiveIntegerField(default=0)

    # Position metadata — crucial for citation rendering
    chunk_index = models.PositiveIntegerField()  # position in document (0-based)
    page_start = models.PositiveIntegerField(default=0)
    page_end = models.PositiveIntegerField(default=0)
    char_start = models.PositiveIntegerField(default=0)  # character offset in raw text
    char_end = models.PositiveIntegerField(default=0)

    # Embedding
    # Gemini gemini-embedding-001 produces 768-dim vectors.
    # We store the chroma_id so we can look up the vector in ChromaDB.
    chroma_id = models.CharField(max_length=100, unique=True, blank=True)
    embedding_model = models.CharField(
        max_length=100, default="models/gemini-embedding-001"
    )
    is_embedded = models.BooleanField(default=False)
    embedded_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "rag_chunk"
        ordering = ["document", "chunk_index"]
        unique_together = [("document", "chunk_index")]
        verbose_name = "Chunk"
        verbose_name_plural = "Chunks"

    def __str__(self):
        return (
            f"Chunk {self.chunk_index} | "
            f"Doc: {self.document.title} | "
            f"Pages {self.page_start}-{self.page_end}"
        )

    def save(self, *args, **kwargs):
        # Auto-generate a stable chroma_id before first save
        if not self.chroma_id:
            self.chroma_id = f"chunk_{self.document_id}_{self.chunk_index}"
        super().save(*args, **kwargs)

    @property
    def preview(self):
        """Short preview for admin / debug views."""
        return self.text[:120] + "..." if len(self.text) > 120 else self.text


class ChatSession(models.Model):

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="chat_sessions",
        null=True,
        blank=True,
    )

    title = models.CharField(
        max_length=255,
        blank=True,
        default="",
        help_text="Auto-generated from first message or set by user.",
    )

    # Which documents are in scope for this session.
    # Empty = search across ALL user documents.
    documents = models.ManyToManyField(
        Document,
        blank=True,
        related_name="chat_sessions",
        help_text="Documents to restrict RAG retrieval to. Empty = all documents.",
    )

    # RAG retrieval config per-session (can override global defaults)
    top_k_retrieval = models.PositiveSmallIntegerField(
        default=20,
        help_text="How many chunks to fetch from ChromaDB before re-ranking.",
    )
    top_k_rerank = models.PositiveSmallIntegerField(
        default=5,
        help_text="How many chunks to pass to Gemini after re-ranking.",
    )

    # Gemini model used for this session
    llm_model = models.CharField(
        max_length=100,
        default="gemini-1.5-flash",
    )

    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "rag_chat_session"
        ordering = ["-updated_at"]
        verbose_name = "Chat Session"
        verbose_name_plural = "Chat Sessions"

    def __str__(self):
        owner = self.user.username if self.user else "anonymous"
        return f"Session [{owner}] — {self.title or self.id}"

    @property
    def message_count(self):
        return self.messages.count()

    def get_history_for_prompt(self, last_n=10):
        """
        Returns the last N message pairs formatted for the Gemini
        multi-turn `contents` parameter.

        Returns a list like:
            [
                {"role": "user",  "parts": [{"text": "..."}]},
                {"role": "model", "parts": [{"text": "..."}]},
                ...
            ]
        """
        recent = self.messages.order_by("-created_at")[: last_n * 2]
        history = []
        for msg in reversed(recent):
            role = "user" if msg.role == Message.Role.USER else "model"
            history.append(
                {
                    "role": role,
                    "parts": [{"text": msg.content}],
                }
            )
        return history


class Message(models.Model):

    class Role(models.TextChoices):
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"
        SYSTEM = "system", "System"

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="messages",
    )

    role = models.CharField(max_length=20, choices=Role.choices)
    content = models.TextField()

    # Which chunks were retrieved and used to build this answer.
    # Populated only for ASSISTANT messages.
    source_chunks = models.ManyToManyField(
        Chunk,
        blank=True,
        related_name="cited_in_messages",
        help_text="Chunks retrieved and used to generate this response.",
    )

    # Gemini usage metadata (tokens billed)
    prompt_tokens = models.PositiveIntegerField(default=0)
    completion_tokens = models.PositiveIntegerField(default=0)
    total_tokens = models.PositiveIntegerField(default=0)

    # Retrieval metadata (for debugging / analytics)
    retrieval_scores = models.JSONField(
        default=list,
        blank=True,
        help_text="List of {chunk_id, similarity_score, rerank_score} dicts.",
    )

    # Was this response streamed to the client?
    was_streamed = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "rag_message"
        ordering = ["session", "created_at"]
        verbose_name = "Message"
        verbose_name_plural = "Messages"

    def __str__(self):
        preview = self.content[:60] + "..." if len(self.content) > 60 else self.content
        return f"[{self.role.upper()}] {preview}"

    @property
    def is_user(self):
        return self.role == self.Role.USER

    @property
    def is_assistant(self):
        return self.role == self.Role.ASSISTANT

    @property
    def cited_documents(self):
        """Distinct documents referenced in this message's source chunks."""
        return Document.objects.filter(chunks__cited_in_messages=self).distinct()
