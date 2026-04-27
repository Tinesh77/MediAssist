"""
management/commands/chunk_document.py
──────────────────────────────────────────────────────────────────
Django management command to manually trigger the chunk pipeline
for one or all pending documents.

Usage:
    # Process a specific document by UUID:
    python manage.py chunk_document 3f2a1b4c-...

    # Process ALL documents with status=pending:
    python manage.py chunk_document --all

    # Force re-process an already-ready document:
    python manage.py chunk_document 3f2a1b4c-... --force

    # Use a specific chunking strategy (overrides doc setting):
    python manage.py chunk_document 3f2a1b4c-... --strategy semantic
"""

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Run the PDF extraction + chunking pipeline for one or all documents."

    def add_arguments(self, parser):
        parser.add_argument(
            "doc_id",
            nargs   = "?",
            type    = str,
            help    = "UUID of the document to process.",
        )
        parser.add_argument(
            "--all",
            action  = "store_true",
            help    = "Process all documents with status=pending.",
        )
        parser.add_argument(
            "--force",
            action  = "store_true",
            help    = "Re-process even if status=ready.",
        )
        parser.add_argument(
            "--strategy",
            choices = ["fixed", "recursive", "semantic"],
            help    = "Override the document's chunk_strategy field.",
        )

    def handle(self, *args, **options):
        from Assistant.models import Document
        from Assistant.services.chunk_pipeline import run_pipeline

        docs = []

        if options["all"]:
            docs = list(Document.objects.filter(status="pending"))
            if not docs:
                self.stdout.write(self.style.WARNING("No pending documents found."))
                return
            self.stdout.write(f"Processing {len(docs)} pending documents...")

        elif options["doc_id"]:
            try:
                doc = Document.objects.get(id=options["doc_id"])
                docs = [doc]
            except Document.DoesNotExist:
                raise CommandError(f"Document '{options['doc_id']}' not found.")

        else:
            raise CommandError("Provide a doc_id or use --all.")

        for doc in docs:
            # Apply strategy override
            if options["strategy"]:
                doc.chunk_strategy = options["strategy"]
                doc.save(update_fields=["chunk_strategy"])

            # Reset status if forcing re-process
            if options["force"] and doc.status == "ready":
                doc.status = "pending"
                doc.save(update_fields=["status"])

            self.stdout.write(f"  Processing: {doc.title} [{doc.chunk_strategy}]")

            try:
                summary = run_pipeline(doc)
                self.stdout.write(self.style.SUCCESS(f"  OK  {summary}"))
            except Exception as exc:
                self.stderr.write(self.style.ERROR(f"  FAIL  {doc.title}: {exc}"))