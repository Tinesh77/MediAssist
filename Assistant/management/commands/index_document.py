"""
management/commands/index_document.py
──────────────────────────────────────────────────────────────────
Django management command to embed and index document chunks
into ChromaDB.

Usage:
    # Index a specific document:
    python manage.py index_document <uuid>

    # Index all ready (chunked) but not yet embedded documents:
    python manage.py index_document --all

    # Force re-index even if already embedded:
    python manage.py index_document <uuid> --force

    # Check ChromaDB stats without indexing:
    python manage.py index_document --stats
"""

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Embed document chunks and store vectors in ChromaDB."

    def add_arguments(self, parser):
        parser.add_argument(
            "doc_id",
            nargs  = "?",
            type   = str,
            help   = "UUID of the document to index.",
        )
        parser.add_argument(
            "--all",
            action = "store_true",
            help   = "Index all ready documents that have unembedded chunks.",
        )
        parser.add_argument(
            "--force",
            action = "store_true",
            help   = "Re-index even if already embedded.",
        )
        parser.add_argument(
            "--stats",
            action = "store_true",
            help   = "Print ChromaDB collection stats and exit.",
        )

    def handle(self, *args, **options):
        from Assistant.models import Document
        from Assistant.services.retrieval_service import (
            ChromaService,
            IndexingService,
        )

        # ── Stats mode ────────────────────────────────────────────
        if options["stats"]:
            total = ChromaService.collection_count()
            self.stdout.write(f"\nChromaDB collection stats:")
            self.stdout.write(f"  Total vectors : {total}")

            docs = Document.objects.filter(status="ready")
            for doc in docs:
                count = ChromaService.document_vector_count(str(doc.id))
                embedded = doc.chunks.filter(is_embedded=True).count()
                total_c  = doc.chunk_count
                self.stdout.write(
                    f"  {doc.title[:40]:<40} "
                    f"vectors={count}  chunks={embedded}/{total_c}"
                )
            return

        # ── Gather documents to index ─────────────────────────────
        docs = []

        if options["all"]:
            # Find ready documents with at least one unembedded chunk
            all_ready = Document.objects.filter(status="ready")
            for doc in all_ready:
                has_unembedded = doc.chunks.filter(is_embedded=False).exists()
                if has_unembedded or options["force"]:
                    docs.append(doc)

            if not docs:
                self.stdout.write(self.style.WARNING(
                    "All ready documents are already indexed. "
                    "Use --force to re-index."
                ))
                return
            self.stdout.write(f"Indexing {len(docs)} documents...")

        elif options["doc_id"]:
            try:
                doc = Document.objects.get(id=options["doc_id"])
                docs = [doc]
            except Document.DoesNotExist:
                raise CommandError(f"Document '{options['doc_id']}' not found.")
        else:
            raise CommandError("Provide a doc_id or use --all or --stats.")

        # ── Index each document ───────────────────────────────────
        for doc in docs:
            if doc.status != "ready":
                self.stderr.write(self.style.WARNING(
                    f"  SKIP  '{doc.title}' — status is '{doc.status}', "
                    f"must be 'ready'. Run chunk_document first."
                ))
                continue

            self.stdout.write(f"  Indexing: {doc.title} ({doc.chunk_count} chunks)...")

            try:
                count = IndexingService.index_document(doc)
                self.stdout.write(self.style.SUCCESS(
                    f"  OK    {count} vectors indexed for '{doc.title}'"
                ))
            except Exception as exc:
                self.stderr.write(self.style.ERROR(
                    f"  FAIL  '{doc.title}': {exc}"
                ))

        # ── Final stats ───────────────────────────────────────────
        total = ChromaService.collection_count()
        self.stdout.write(f"\nChromaDB total vectors: {total}")