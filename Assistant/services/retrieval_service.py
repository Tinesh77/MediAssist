"""
services/retrieval_service.py
──────────────────────────────────────────────────────────────────
RetrievalService — Phase 3 of MediAssist RAG pipeline.

Responsibilities:
  1. EmbeddingService   — embed texts via Gemini gemini-embedding-001
  2. ChromaService      — manage ChromaDB collections, upsert & delete vectors
  3. IndexingService    — embed all Chunks of a Document and store in Chroma
  4. RetrievalService   — embed a user query and return the top-k most
                          semantically similar Chunk objects from the DB

Full pipeline flow:
    Upload PDF
        → ChunkPipeline  (Phase 2)  → Chunk rows in PostgreSQL
        → IndexingService(Phase 3)  → vectors in ChromaDB
        → RetrievalService(Phase 3) → top-k Chunk objects for a query
        → PromptBuilder  (Phase 4)  → Gemini answer with citations

Usage:
    # After chunking a document, index it:
    from .services.retrieval_service import IndexingService, RetrievalService

    IndexingService.index_document(document)          # stores vectors in Chroma

    # At query time:
    results = RetrievalService.search(
        query      = "What are the symptoms of hypertension?",
        document   = document,          # or pass document_ids=[...] for multi-doc
        top_k      = 5,
    )
    for r in results:
        print(r.score, r.chunk.text[:80])

Standalone test (no Django):
    python services/retrieval_service.py
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
#  Config — reads from environment / Django settings
# ─────────────────────────────────────────────────────────────────

def _get_chroma_path() -> str:
    """
    Where ChromaDB stores its persistent vector index on disk.
    Reads CHROMA_DB_PATH from env, falls back to ./chroma_store/
    """
    try:
        from django.conf import settings
        return getattr(settings, "CHROMA_DB_PATH", "chroma_store/")
    except Exception:
        return os.environ.get("CHROMA_DB_PATH", "chroma_store/")


def _get_gemini_key() -> str:
    try:
        from django.conf import settings
        return getattr(settings, "GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
    except Exception:
        return os.environ.get("GEMINI_API_KEY", "")


EMBEDDING_MODEL  = "gemini-embedding-001"
EMBEDDING_DIM    = 768   # Gemini embedding-001 target output dimension
EMBED_BATCH_SIZE = 50    # chunks per API batch (Gemini free-tier safe)
EMBED_SLEEP_SEC  = 1.0   # sleep between batches

# ── Collection strategy ───────────────────────────────────────────
# Controls how ChromaDB collections are organised.
#
#   "global"       — one collection "mediassist" for the whole app.
#                  Simplest. Filter by document_id at query time.
#                  Best for: small apps, single-user projects.
#
#   "per_document" — one collection per Document UUID.
#                  e.g. "doc_3f2a1b4c_8d9e_..."
#                  No metadata filter needed at query time.
#                  Best for: strict document isolation, large docs.
#
#   "per_user"   — one collection per User ID.
#                  e.g. "user_42"
#                  All documents for a user share one collection.
#                  Filter by document_id within the collection.
#                  Best for: multi-tenant apps, data separation.
#
# Set CHROMA_COLLECTION_STRATEGY in your .env or settings.py.
# Defaults to "per_document" — cleanest for a learning project.

def _get_strategy() -> str:
    try:
        from django.conf import settings
        return getattr(settings, "CHROMA_COLLECTION_STRATEGY", "per_document")
    except Exception:
        return os.environ.get("CHROMA_COLLECTION_STRATEGY", "per_document")


def _collection_name(document=None, user_id: int = None) -> str:
    """
    Return the ChromaDB collection name based on the active strategy.
    """
    strategy = _get_strategy()

    if strategy == "per_document" and document is not None:
        # Strip hyphens — ChromaDB collection names must be alphanumeric + underscores
        safe_id = str(document.id).replace("-", "")
        return f"doc_{safe_id}"

    if strategy == "per_user" and user_id is not None:
        return f"user_{user_id}"

    # Default / global fallback
    return "mediassist"


# ─────────────────────────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """
    One result from a semantic search.
    """
    chunk:     object
    score:     float
    chroma_id: str
    metadata:  dict = field(default_factory=dict)

    @property
    def page(self) -> int:
        return self.chunk.page_start

    @property
    def preview(self) -> str:
        t = self.chunk.text
        return t[:120] + "..." if len(t) > 120 else t

    @property
    def source_label(self) -> str:
        """
        Citation-ready label combining document title + page range.
        """
        title = getattr(self.chunk.document, "title", "Unknown document")
        ps    = self.chunk.page_start
        pe    = self.chunk.page_end
        page  = f"p.{ps}" if ps == pe else f"p.{ps}–{pe}"
        return f"{title} · {page}"


# ─────────────────────────────────────────────────────────────────
#  EmbeddingService
# ─────────────────────────────────────────────────────────────────

class EmbeddingService:
    """
    Thin wrapper around the modern google-genai embedding API.
    """

    _client = None

    @classmethod
    def _ensure_init(cls):
        if cls._client is None:
            from google import genai
            key = _get_gemini_key()
            if not key:
                raise EnvironmentError(
                    "GEMINI_API_KEY is not set. "
                    "Add it to your .env file."
                )
            cls._client = genai.Client(api_key=key)

    @classmethod
    def embed_text(cls, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        """
        Embed a single piece of text. Returns a 768-dim float list.
        """
        cls._ensure_init()
        from google.genai import types

        result = cls._client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type.upper(),
                output_dimensionality=EMBEDDING_DIM
            )
        )
        return result.embeddings[0].values

    @classmethod
    def embed_query(cls, query: str) -> list[float]:
        """Convenience: embed a user query with the correct task type."""
        return cls.embed_text(query, task_type="RETRIEVAL_QUERY")

    @classmethod
    def embed_chunks_batch(cls, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of chunk texts in rate-limit-safe batches.
        """
        cls._ensure_init()
        from google.genai import types

        all_embeddings = []
        total = len(texts)

        for batch_start in range(0, total, EMBED_BATCH_SIZE):
            batch = texts[batch_start : batch_start + EMBED_BATCH_SIZE]
            logger.info(
                f"[EmbeddingService] Embedding batch "
                f"{batch_start // EMBED_BATCH_SIZE + 1}/"
                f"{(total + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE} "
                f"({len(batch)} chunks)"
            )

            # Send the entire batch list in one API call
            result = cls._client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIM
                )
            )
            
            batch_embeddings = [e.values for e in result.embeddings]
            all_embeddings.extend(batch_embeddings)

            # Rate-limit pause — skip on last batch
            if batch_start + EMBED_BATCH_SIZE < total:
                time.sleep(EMBED_SLEEP_SEC)

        return all_embeddings


# ─────────────────────────────────────────────────────────────────
#  ChromaService
# ─────────────────────────────────────────────────────────────────

class ChromaService:
    """
    Manages ChromaDB persistent client and named collections.
    """

    _client      = None
    _collections: dict = {}   # cache: name → collection object

    @classmethod
    def _get_client(cls):
        """Lazily initialise the persistent ChromaDB client (singleton)."""
        if cls._client is not None:
            return cls._client

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required.\n"
                "Install:  pip install chromadb"
            )

        chroma_path = _get_chroma_path()
        Path(chroma_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"[ChromaService] Persistent client → '{chroma_path}'")

        cls._client = chromadb.PersistentClient(
            path     = chroma_path,
            settings = Settings(anonymized_telemetry=False),
        )
        return cls._client

    @classmethod
    def get_collection(cls, name: str):
        """
        Get or create a named ChromaDB collection.
        Results are cached so repeated calls within a request are free.
        """
        if name not in cls._collections:
            client = cls._get_client()
            col    = client.get_or_create_collection(
                name     = name,
                metadata = {"hnsw:space": "cosine"},
            )
            cls._collections[name] = col
            logger.info(
                f"[ChromaService] Collection '{name}' ready "
                f"({col.count()} vectors, strategy={_get_strategy()})"
            )
        return cls._collections[name]

    @classmethod
    def get_collection_for(cls, document=None, user_id: int = None):
        """
        Convenience: resolve collection name from document/user,
        then return the collection object.
        """
        name = _collection_name(document=document, user_id=user_id)
        return cls.get_collection(name)

    # ── Upsert ────────────────────────────────────────────────────

    @classmethod
    def upsert(
        cls,
        ids:             list[str],
        embeddings:      list[list[float]],
        documents:       list[str],
        metadatas:       list[dict],
        collection_name: str = "mediassist",
    ) -> None:
        """
        Insert or update vectors in a named collection.
        """
        collection = cls.get_collection(collection_name)
        batch_size = 500

        for i in range(0, len(ids), batch_size):
            collection.upsert(
                ids        = ids[i : i + batch_size],
                embeddings = embeddings[i : i + batch_size],
                documents  = documents[i : i + batch_size],
                metadatas  = metadatas[i : i + batch_size],
            )
        logger.info(
            f"[ChromaService] Upserted {len(ids)} vectors "
            f"→ collection '{collection_name}'"
        )

    # ── Query ─────────────────────────────────────────────────────

    @classmethod
    def query(
        cls,
        query_embedding:  list[float],
        top_k:            int,
        collection_name:  str = "mediassist",
        where:            Optional[dict] = None,
    ) -> dict:
        """
        Find the top_k most similar vectors in a named collection.
        """
        collection = cls.get_collection(collection_name)
        count      = collection.count()

        if count == 0:
            return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

        kwargs = dict(
            query_embeddings = [query_embedding],
            n_results        = min(top_k, count),
            include          = ["documents", "distances", "metadatas"],
        )
        if where:
            kwargs["where"] = where

        return collection.query(**kwargs)

    # ── Delete ────────────────────────────────────────────────────

    @classmethod
    def delete_document(cls, document_id: str, collection_name: str = "mediassist") -> int:
        """
        Remove all vectors for a document from its collection.
        """
        strategy = _get_strategy()

        if strategy == "per_document":
            client = cls._get_client()
            try:
                client.delete_collection(collection_name)
                cls._collections.pop(collection_name, None)
                logger.info(f"[ChromaService] Dropped collection '{collection_name}'")
                return -1   # count unknown after drop
            except Exception as e:
                logger.warning(f"[ChromaService] Could not drop collection: {e}")
                return 0

        collection   = cls.get_collection(collection_name)
        existing     = collection.get(
            where   = {"document_id": {"$eq": document_id}},
            include = [],
        )
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            logger.info(
                f"[ChromaService] Deleted {len(ids_to_delete)} vectors "
                f"for document {document_id} from '{collection_name}'"
            )
        return len(ids_to_delete)

    # ── Stats ─────────────────────────────────────────────────────

    @classmethod
    def collection_count(cls, collection_name: str = "mediassist") -> int:
        try:
            return cls.get_collection(collection_name).count()
        except Exception:
            return 0

    @classmethod
    def document_vector_count(cls, document_id: str, collection_name: str = "mediassist") -> int:
        strategy = _get_strategy()

        if strategy == "per_document":
            return cls.collection_count(collection_name)

        collection = cls.get_collection(collection_name)
        result     = collection.get(
            where   = {"document_id": {"$eq": document_id}},
            include = [],
        )
        return len(result.get("ids", []))


# ─────────────────────────────────────────────────────────────────
#  IndexingService
# ─────────────────────────────────────────────────────────────────

class IndexingService:
    """
    Takes a Document's Chunk rows from PostgreSQL,
    embeds each chunk via Gemini, and stores the vectors in ChromaDB.
    """

    @classmethod
    def index_document(cls, document, user_id: int = None) -> int:
        from django.utils import timezone
        from ..models import Chunk

        chunks = list(document.chunks.order_by("chunk_index"))

        if not chunks:
            raise ValueError(
                f"Document '{document.title}' has no chunks. "
                f"Run chunk_document first."
            )

        collection_name = _collection_name(document=document, user_id=user_id)
        strategy        = _get_strategy()

        logger.info(
            f"[IndexingService] Indexing '{document.title}' — "
            f"{len(chunks)} chunks → collection '{collection_name}' "
            f"(strategy={strategy})"
        )

        texts      = [c.text for c in chunks]
        embeddings = EmbeddingService.embed_chunks_batch(texts)

        ids       = [c.chroma_id for c in chunks]
        metadatas = []

        for chunk in chunks:
            metadatas.append({
                "document_id": str(document.id),
                "chunk_index": chunk.chunk_index,
                "page_start":  chunk.page_start,
                "page_end":    chunk.page_end,
                "char_start":  chunk.char_start,
                "char_end":    chunk.char_end,
                "token_count": chunk.token_count,
                "doc_title":   document.title,
                "user_id":     str(user_id) if user_id else "",
            })

        ChromaService.upsert(
            ids             = ids,
            embeddings      = embeddings,
            documents       = texts,
            metadatas       = metadatas,
            collection_name = collection_name,
        )

        Chunk.objects.filter(
            id__in = [c.id for c in chunks]
        ).update(
            is_embedded     = True,
            embedded_at     = timezone.now(),
            embedding_model = EMBEDDING_MODEL,
        )

        total = ChromaService.collection_count(collection_name)
        logger.info(
            f"[IndexingService] Done — {len(chunks)} chunks indexed. "
            f"Collection '{collection_name}' now has {total} vectors."
        )
        return len(chunks)

    @classmethod
    def index_single_chunk(cls, chunk, user_id: int = None) -> None:
        from django.utils import timezone
        from ..models import Chunk

        collection_name = _collection_name(
            document = chunk.document,
            user_id  = user_id,
        )
        embedding = EmbeddingService.embed_text(chunk.text)

        ChromaService.upsert(
            ids             = [chunk.chroma_id],
            embeddings      = [embedding],
            documents       = [chunk.text],
            metadatas       = [{
                "document_id": str(chunk.document_id),
                "chunk_index": chunk.chunk_index,
                "page_start":  chunk.page_start,
                "page_end":    chunk.page_end,
                "char_start":  chunk.char_start,
                "char_end":    chunk.char_end,
                "token_count": chunk.token_count,
                "doc_title":   chunk.document.title,
                "user_id":     str(user_id) if user_id else "",
            }],
            collection_name = collection_name,
        )

        Chunk.objects.filter(id=chunk.id).update(
            is_embedded     = True,
            embedded_at     = timezone.now(),
            embedding_model = EMBEDDING_MODEL,
        )

    @classmethod
    def deindex_document(cls, document, user_id: int = None) -> int:
        collection_name = _collection_name(document=document, user_id=user_id)
        deleted = ChromaService.delete_document(
            document_id     = str(document.id),
            collection_name = collection_name,
        )
        logger.info(
            f"[IndexingService] Deindexed '{document.title}' "
            f"from '{collection_name}'"
        )
        return deleted


# ─────────────────────────────────────────────────────────────────
#  RetrievalService
# ─────────────────────────────────────────────────────────────────

class RetrievalService:
    """
    The semantic search engine for the RAG pipeline.
    """

    @classmethod
    def search(
        cls,
        query:        str,
        top_k:        int       = 20,
        document:     object    = None,
        document_ids: list[str] = None,
        user_id:      int       = None,
    ) -> list[SearchResult]:
        strategy = _get_strategy()

        logger.info(f"[RetrievalService] Query: '{query[:70]}'")
        query_embedding = EmbeddingService.embed_query(query)

        collection_name, where = cls._resolve_collection(
            document     = document,
            document_ids = document_ids,
            user_id      = user_id,
            strategy     = strategy,
        )

        count = ChromaService.collection_count(collection_name)
        if count == 0:
            logger.warning(
                f"[RetrievalService] Collection '{collection_name}' is empty. "
                f"Run IndexingService.index_document() first."
            )
            return []

        logger.info(
            f"[RetrievalService] Searching top_k={top_k} in "
            f"'{collection_name}' ({count} vectors)"
            + (f" | filter={where}" if where else "")
        )

        raw = ChromaService.query(
            query_embedding  = query_embedding,
            top_k            = top_k,
            collection_name  = collection_name,
            where            = where,
        )

        chroma_ids = raw["ids"][0]
        distances  = raw["distances"][0]
        metadatas  = raw["metadatas"][0]

        if not chroma_ids:
            logger.info("[RetrievalService] No results returned.")
            return []

        scores = [round(1 - (d / 2), 4) for d in distances]

        results = cls._fetch_and_build(chroma_ids, scores, metadatas)

        logger.info(
            f"[RetrievalService] Returned {len(results)} results "
            f"| top score={results[0].score if results else 'n/a'}"
        )
        return results

    # ── Internal: collection + filter resolution ──────────────────

    @classmethod
    def _resolve_collection(
        cls,
        document,
        document_ids,
        user_id,
        strategy,
    ) -> tuple[str, Optional[dict]]:
        if strategy == "per_document":
            if document is not None:
                return _collection_name(document=document), None
            logger.warning(
                "[RetrievalService] per_document strategy with multiple docs: "
                "searching first document's collection only. "
                "Switch to 'global' strategy for multi-doc search."
            )
            return "mediassist", None

        if strategy == "per_user":
            col_name = _collection_name(user_id=user_id)
            where    = cls._build_where(document, document_ids)
            return col_name, where

        where = cls._build_where(document, document_ids)
        return "mediassist", where

    @classmethod
    def _build_where(cls, document, document_ids) -> Optional[dict]:
        if document is not None:
            return {"document_id": {"$eq": str(document.id)}}
        if document_ids:
            if len(document_ids) == 1:
                return {"document_id": {"$eq": document_ids[0]}}
            return {"document_id": {"$in": document_ids}}
        return None

    # ── Internal: fetch Chunk rows + build SearchResult list ───────

    @classmethod
    def _fetch_and_build(
        cls,
        chroma_ids: list[str],
        scores:     list[float],
        metadatas:  list[dict],
    ) -> list[SearchResult]:
        from ..models import Chunk

        chunks_by_id = {
            c.chroma_id: c
            for c in Chunk.objects.filter(
                chroma_id__in = chroma_ids
            ).select_related("document")
        }

        results = []
        for chroma_id, score, meta in zip(chroma_ids, scores, metadatas):
            chunk = chunks_by_id.get(chroma_id)
            if chunk is None:
                logger.warning(
                    f"[RetrievalService] '{chroma_id}' in ChromaDB but not "
                    f"in PostgreSQL — re-index the document to fix."
                )
                continue

            results.append(SearchResult(
                chunk     = chunk,
                score     = score,
                chroma_id = chroma_id,
                metadata  = meta,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results


# ─────────────────────────────────────────────────────────────────
#  Django signal — auto-deindex on Document delete
# ─────────────────────────────────────────────────────────────────
#
# Add this to your app's apps.py ready() method, or signals.py:
#
#   from django.db.models.signals import post_delete
#   from django.dispatch import receiver
#   from .models import Document
#   from .services.retrieval_service import IndexingService
#
#   @receiver(post_delete, sender=Document)
#   def deindex_on_delete(sender, instance, **kwargs):
#       try:
#           IndexingService.deindex_document(instance)
#       except Exception as e:
#           logger.warning(f"Failed to deindex document: {e}")


# ─────────────────────────────────────────────────────────────────
#  Standalone smoke test
#  Run: python services/retrieval_service.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("  RetrievalService — Standalone Smoke Test")
    print("=" * 60)

    # ── Test 1: Embedding ─────────────────────────────────────────
    print("\n[1] EmbeddingService — embed single text ...")
    try:
        vec = EmbeddingService.embed_text(
            "Hypertension is a chronic condition characterised by elevated blood pressure.",
            task_type="RETRIEVAL_DOCUMENT",
        )
        print(f"    [OK] Vector dim: {len(vec)}")
        print(f"         First 5:  {[round(v, 6) for v in vec[:5]]}")
    except Exception as e:
        print(f"    [FAIL] {e}")
        sys.exit(1)

    # ── Test 2: Query embedding ───────────────────────────────────
    print("\n[2] EmbeddingService — embed query ...")