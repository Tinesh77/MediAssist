
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Chunk and embed a document by UUID"

    def add_arguments(self, parser):
        parser.add_argument("doc_id", type=str)

    def handle(self, *args, **options):
        from Assistant.services.chunking import ChunkingService
        from Assistant.models import Document
        doc     = Document.objects.get(id=options["doc_id"])
        service = ChunkingService(doc)
        chunks  = service.run_and_save()
        self.stdout.write(f"Created {len(chunks)} chunks.")