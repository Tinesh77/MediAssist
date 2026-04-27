"""
management/commands/ingest_pdfs.py
──────────────────────────────────────────────────────────────────
Bulk-ingest a folder of medical PDFs into MediAssist at startup.

Runs the full pipeline for each PDF:
    Upload → Chunk → Embed → Index into ChromaDB

Usage:
    # Ingest all PDFs from a folder:
    python manage.py ingest_pdfs /path/to/medical_docs/

    # Preview what would be ingested (dry run):
    python manage.py ingest_pdfs /path/to/docs/ --dry-run

    # Use a specific chunking strategy:
    python manage.py ingest_pdfs /path/to/docs/ --strategy recursive

    # Skip already-processed documents (idempotent):
    python manage.py ingest_pdfs /path/to/docs/ --skip-existing

    # Force re-process everything:
    python manage.py ingest_pdfs /path/to/docs/ --force

Example startup script (ingest_startup.sh):
    #!/bin/bash
    python manage.py migrate
    python manage.py ingest_pdfs /app/medical_docs/ --skip-existing
    gunicorn MediAssist.wsgi:application --bind 0.0.0.0:8000
"""

import os
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Bulk-ingest PDFs from a directory — extract, chunk, embed, index."

    def add_arguments(self, parser):
        parser.add_argument(
            "folder",
            type=str,
            help="Path to folder containing PDF files.",
        )
        parser.add_argument(
            "--strategy",
            choices=["fixed", "recursive", "semantic"],
            default="recursive",
            help="Chunking strategy (default: recursive).",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=512,
            help="Chunk size in tokens (default: 512).",
        )
        parser.add_argument(
            "--chunk-overlap",
            type=int,
            default=64,
            help="Chunk overlap in tokens (default: 64).",
        )
        parser.add_argument(
            "--skip-existing",
            action="store_true",
            help="Skip PDFs that are already in the database.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-process documents even if already ready.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print what would be ingested without doing it.",
        )
        parser.add_argument(
            "--no-index",
            action="store_true",
            help="Only chunk — skip embedding and ChromaDB indexing.",
        )

    def handle(self, *args, **options):
        from Assistant.models import Document
        from Assistant.services.chunk_pipeline import run_pipeline
        from Assistant.services.retrieval_service import IndexingService

        folder = Path(options["folder"])
        if not folder.exists() or not folder.is_dir():
            raise CommandError(f"Folder not found: {folder}")

        # Collect all PDFs recursively
        pdfs = sorted(folder.rglob("*.pdf"))
        if not pdfs:
            self.stdout.write(self.style.WARNING(f"No PDF files found in {folder}"))
            return

        self.stdout.write(f"\nFound {len(pdfs)} PDF file(s) in {folder}\n")

        # ── Dry run ───────────────────────────────────────────────
        if options["dry_run"]:
            for pdf in pdfs:
                exists = Document.objects.filter(
                    title=self._pdf_to_title(pdf)
                ).exists()
                flag = "[EXISTS]" if exists else "[NEW]   "
                size = pdf.stat().st_size // 1024
                self.stdout.write(f"  {flag}  {pdf.name} ({size} KB)")
            return

        # ── Ingest loop ───────────────────────────────────────────
        success_count = 0
        skip_count    = 0
        fail_count    = 0

        for pdf in pdfs:
            title = self._pdf_to_title(pdf)
            self.stdout.write(f"\n  Processing: {pdf.name}")

            # Check if already exists
            existing = Document.objects.filter(title=title).first()

            if existing:
                if options["skip_existing"] and existing.status == "ready":
                    self.stdout.write(
                        self.style.WARNING(f"    SKIP  Already ready ({existing.chunk_count} chunks)")
                    )
                    skip_count += 1
                    continue
                elif not options["force"] and existing.status == "ready":
                    self.stdout.write(
                        self.style.WARNING(f"    SKIP  Already processed. Use --force to re-process.")
                    )
                    skip_count += 1
                    continue
                else:
                    # Force re-process — reset status
                    doc = existing
                    doc.status = "pending"
                    doc.save(update_fields=["status"])
            else:
                # Create new Document record
                doc = self._create_document(pdf, title, options)

            # ── Step 1: Chunk ─────────────────────────────────────
            try:
                self.stdout.write(f"    Chunking ({options['strategy']})...")
                summary = run_pipeline(doc)
                self.stdout.write(self.style.SUCCESS(f"    {summary}"))
            except Exception as exc:
                self.stderr.write(self.style.ERROR(f"    FAIL  Chunking: {exc}"))
                fail_count += 1
                continue

            # ── Step 2: Index into ChromaDB ───────────────────────
            if not options["no_index"]:
                try:
                    self.stdout.write(f"    Embedding + indexing...")
                    count = IndexingService.index_document(doc)
                    self.stdout.write(
                        self.style.SUCCESS(f"    OK  {count} vectors indexed")
                    )
                except Exception as exc:
                    self.stderr.write(
                        self.style.ERROR(f"    FAIL  Indexing: {exc}")
                    )
                    fail_count += 1
                    continue

            success_count += 1

        # ── Summary ───────────────────────────────────────────────
        self.stdout.write("\n" + "─" * 50)
        self.stdout.write(
            f"  Ingested: {success_count}  |  "
            f"Skipped: {skip_count}  |  "
            f"Failed: {fail_count}"
        )
        self.stdout.write("─" * 50 + "\n")

    def _pdf_to_title(self, pdf_path: Path) -> str:
        """Convert filename to a clean document title."""
        return (
            pdf_path.stem
            .replace("_", " ")
            .replace("-", " ")
            .title()
        )

    def _create_document(self, pdf_path: Path, title: str, options: dict):
        """Create a Document record from a local PDF file."""
        from django.core.files import File
        from Assistant.models import Document

        size_kb = pdf_path.stat().st_size // 1024

        with open(pdf_path, "rb") as f:
            django_file = File(f, name=pdf_path.name)
            doc = Document(
                title          = title,
                file_type      = "pdf",
                file_size_kb   = size_kb,
                chunk_strategy = options["strategy"],
                chunk_size     = options["chunk_size"],
                chunk_overlap  = options["chunk_overlap"],
                status         = "pending",
            )
            doc.file.save(pdf_path.name, django_file, save=True)

        self.stdout.write(f"    Created document record (id={doc.id})")
        return doc