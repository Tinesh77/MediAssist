# MediAssist — Healthcare RAG Assistant

> **Ask anything about your health. Get answers from your records.**

MediAssist is a production-grade, end-to-end **Retrieval-Augmented Generation (RAG)** system built for healthcare documents. Upload medical PDFs — lab reports, clinical notes, drug guides, research papers — and ask questions in plain English. Every answer is grounded in your documents, streamed in real time, and backed by clickable source citations linked to the exact page.

---

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [RAG Pipeline](#rag-pipeline)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Author](#author)

---

## Demo

| Chat with citations | Document Library | Chunk Inspector |
|---|---|---|
| Streaming answers with `[Source N]` chips | Upload, manage & delete PDFs | Debug chunks, embeddings, vector search |

---

## Features

### Core RAG
- **Multi-strategy chunking** — Fixed-size, Recursive (paragraph → sentence → word), and Semantic (embedding-based topic boundary detection)
- **Gemini Embeddings** — `gemini-embedding-001` (768-dim vectors) with `retrieval_document` / `retrieval_query` task types for optimised retrieval
- **ChromaDB vector store** — Persistent local storage with per-document, per-user, or global collection strategies
- **Cross-encoder re-ranking** — `sentence-transformers` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores top-20 ChromaDB candidates → top-5 by true relevance
- **Chain-of-thought prompting** — 3-step CoT (Identify → Synthesise → Answer) with mandatory `[Source N]` citation rules
- **Gemini 1.5 Flash** — Fast, cost-efficient LLM for answer generation

### Document Processing
- **Dual-mode PDF extraction** — `pdfplumber` for selectable text PDFs, automatic OCR fallback (`PyMuPDF` + `pytesseract`) for scanned/image-based documents
- **Table extraction** — Converts PDF tables to pipe-delimited text preserving structure
- **Header/footer removal** — Deduplicates repeated lines across pages
- **Background processing** — Upload returns instantly; chunking + embedding runs in a daemon thread

### Frontend
- **SSE streaming** — Token-by-token streaming via `fetch` + `ReadableStream`
- **Citation chips** — Clickable source chips showing document title, page range, retrieval score, re-rank score, and chunk text
- **Inline citation highlighting** — `[Source N]` tags in the answer are clickable and cross-highlight the corresponding chip
- **Document library** — Upload, browse, and delete documents with real-time status polling
- **Conversation history** — Full session persistence via `ChatSession` + `Message` models
- **Chunk inspector** — Debug UI for inspecting chunks, embeddings, and semantic search results

### Developer Tools
- **Chunk inspector** at `/inspector/` — visualise chunk boundaries, run test searches, trigger re-indexing
- **Management commands** — `chunk_document`, `index_document`, `ingest_pdfs` for bulk operations
- **Bulk ingestion** — `ingest_pdfs` command with `--skip-existing`, `--dry-run`, `--force`, `--strategy` flags

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Django 4.2, Django REST Framework |
| **AI / LLM** | Google Gemini 2.5 Flash (`gemini-1.5-flash`) |
| **Embeddings** | Google Gemini `models/embedding-001` (768-dim) |
| **Vector Store** | ChromaDB (persistent, cosine similarity) |
| **Re-ranking** | sentence-transformers `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **PDF (native)** | pdfplumber |
| **PDF (OCR)** | PyMuPDF + pytesseract + Tesseract binary |
| **Database** | PostgreSQL (production) / SQLite (development) |
| **Frontend** | Vanilla HTML, CSS, JavaScript (no framework) |
| **Streaming** | Server-Sent Events (SSE) via `StreamingHttpResponse` |
| **Auth** | Session-based (pluggable, disabled for local dev) |
| **Deploy** | Gunicorn + Whitenoise |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                             │
│          HTML / CSS / JS  ·  SSE streaming                  │
│   Upload │ Chat UI │ Citation chips │ Document library       │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼──────────────────────────────────────┐
│                   Django Backend                            │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Upload API  │  │  Chat API    │  │  Debug / Search  │   │
│  │ /documents/ │  │ /chat/stream/│  │  /inspector/     │   │
│  └──────┬──────┘  └──────┬───────┘  └──────────────────┘   │
│         │                │                                  │
│  ┌──────▼──────────────────────────────────────────────┐    │
│  │               RAG Pipeline                          │    │
│  │                                                     │    │
│  │  PDFExtractor → ChunkingService → IndexingService   │    │
│  │       ↓               ↓                ↓            │    │
│  │    OCR/text       Chunk rows      ChromaDB vectors  │    │
│  │                                                     │    │
│  │  RetrievalService → ReRanker → PromptBuilder        │    │
│  │       ↓               ↓              ↓              │    │
│  │  top-20 chunks   top-5 chunks    Gemini prompt      │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────┬───────────────────────┬──────────────────────┘
               │                       │
┌──────────────▼──────┐   ┌────────────▼────────────┐
│     PostgreSQL      │   │        ChromaDB          │
│  Documents, Chunks  │   │   768-dim vector index   │
│  Sessions, Messages │   │   per-document collections│
└─────────────────────┘   └──────────────────────────┘
                                      │
                          ┌───────────▼──────────────┐
                          │    Google Gemini API      │
                          │  embedding-001 (vectors)  │
                          │  gemini-2.5-flash (LLM)   │
                          └──────────────────────────┘
```

---

## RAG Pipeline

MediAssist implements a 5-phase RAG pipeline:

### Phase 1 — Document Ingestion
```
Upload PDF → Validate (magic bytes + size) → Save to DB (status=pending)
→ Background thread: extract → chunk → embed → index
```

**PDF Extraction (dual-mode):**
- `pdfplumber` runs first — fast, accurate for selectable text, preserves tables
- If extracted chars < 50 per page → automatic OCR fallback
- OCR: `PyMuPDF` renders page at 2.5× zoom → `pytesseract` reads pixel content
- Handles ligatures, smart quotes, hyphenated line breaks, header/footer deduplication

### Phase 2 — Chunking
Three strategies, switchable per document:

| Strategy | How it works | Best for |
|---|---|---|
| **Fixed-size** | Splits every N tokens with overlap | Structured data, tables |
| **Recursive** | paragraph → sentence → word boundaries | Clinical notes, research papers |
| **Semantic** | Embeds sentences, splits where cosine similarity drops | Long mixed-topic documents |

Each chunk stored in PostgreSQL with `page_start`, `page_end`, `char_start`, `char_end`, `token_count`, `chroma_id`.

### Phase 3 — Embedding & Vector Storage
```
Chunk text → Gemini embed_content(task_type="retrieval_document") → 768-dim vector
→ ChromaDB upsert (collection: doc_{uuid})
→ Chunk.is_embedded = True
```

**Collection strategy** (configurable via `CHROMA_COLLECTION_STRATEGY`):
- `per_document` — one ChromaDB collection per PDF (default, cleanest isolation)
- `per_user` — one collection per user (multi-tenant)
- `global` — single shared collection with metadata filtering

### Phase 4 — Retrieval & Re-ranking
```
User query → embed(task_type="retrieval_query") → ChromaDB top-20
→ CrossEncoder.predict([(query, chunk)] × 20) → normalise scores
→ Sort by re-rank score → take top-5
→ Build citation map: "[Source 1]" → SourceChunk(chroma_id, page, score)
```

**Re-ranking strategy cascade:**
1. `sentence-transformers` CrossEncoder (primary — local, ~50ms, no API cost)
2. Gemini listwise scoring (fallback — one API call for all candidates)
3. Weighted blend: `0.85 × cosine + 0.15 × position` (last resort)

### Phase 5 — Prompt Building & Generation
```
System prompt (role + safety rules + citation rules)
+ Context block ([Source N] headers + chunk text, token-budgeted to 6000 tokens)
+ Chain-of-thought instructions (STEP 1: Identify → STEP 2: Synthesise → STEP 3: Answer)
+ Conversation history (last 6 turns)
+ User question
→ gemini-1.5-flash → stream tokens → SSE → browser
→ Parse [Source N] citations → link to Chunk DB rows → save Message
```

---

## Project Structure

```
MediAssist/
│
├── MediAssist/                  # Django project config
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── Assistant/                   # Main app
│   ├── models.py                # Document, Chunk, ChatSession, Message
│   │
│   ├── services/
│   │   ├── pdf_extractor.py     # pdfplumber + OCR fallback
│   │   ├── chunking.py          # Fixed / Recursive / Semantic chunkers
│   │   ├── chunk_pipeline.py    # Extract → chunk → save pipeline
│   │   ├── retrieval_service.py # EmbeddingService, ChromaService, RetrievalService
│   │   ├── prompt_builder.py    # ReRanker, PromptBuilder, ChatService
│   │   └── gemini.py            # GeminiService wrapper
│   │
│   ├── api/
│   │   ├── views.py             # Document upload/list/delete + background worker
│   │   ├── serializers.py       # DocumentUploadSerializer (validation)
│   │   ├── urls.py              # All API routes
│   │   ├── chat_views.py        # ChatView, ChatStreamView, SessionListView
│   │   ├── retrieval_views.py   # IndexDocumentView, SemanticSearchView, SearchStatsView
│   │   └── debug_views.py       # DebugDocumentListView, DebugChunkListView, DebugRechunkView
│   │
│   └── management/commands/
│       ├── chunk_document.py    # python manage.py chunk_document
│       ├── index_document.py    # python manage.py index_document
│       └── ingest_pdfs.py       # python manage.py ingest_pdfs <folder>
│
├── templates/
│   ├── index.html               # Main chat UI
│   └── chunk_inspector.html     # Debug / retrieval inspector
│
├── chroma_store/                # ChromaDB persistent vector index (auto-created)
├── media/                       # Uploaded PDFs (auto-created)
├── requirements.txt
├── gunicorn.conf.py
├── .env.example
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Tesseract OCR binary (for scanned PDFs)
- Google Gemini API key

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/mediassist.git
cd mediassist
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract binary

**Windows:**
Download and run the installer from:
https://github.com/UB-Mannheim/tesseract/wiki

Then add to `settings.py`:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

**Linux:**
```bash
sudo apt install tesseract-ocr
```

**Mac:**
```bash
brew install tesseract
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:
```env
SECRET_KEY=your-django-secret-key
GEMINI_API_KEY=your-gemini-api-key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
CHROMA_DB_PATH=chroma_store/
CHROMA_COLLECTION_STRATEGY=per_document
```

Get your Gemini API key at: https://aistudio.google.com/app/apikey

### 6. Run migrations

```bash
python manage.py migrate
```

### 7. Start the server

```bash
python manage.py runserver
```

Open http://127.0.0.1:8000

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `SECRET_KEY` | ✅ | — | Django secret key |
| `GEMINI_API_KEY` | ✅ | — | Google Gemini API key |
| `DEBUG` | ❌ | `False` | Django debug mode |
| `ALLOWED_HOSTS` | ❌ | `localhost` | Comma-separated allowed hosts |
| `DATABASE_URL` | ❌ | SQLite | PostgreSQL connection string |
| `CHROMA_DB_PATH` | ❌ | `chroma_store/` | ChromaDB persistence directory |
| `CHROMA_COLLECTION_STRATEGY` | ❌ | `per_document` | `per_document` / `per_user` / `global` |
| `TESSERACT_CMD` | ❌ | system PATH | Full path to tesseract.exe (Windows) |

---

## API Reference

### Documents

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/documents/` | List all documents |
| `POST` | `/api/documents/upload/` | Upload a PDF |
| `GET` | `/api/documents/<id>/` | Get document + chunk preview |
| `DELETE` | `/api/documents/<id>/` | Delete document |
| `POST` | `/api/documents/<id>/index/` | Embed + index into ChromaDB |

### Chat

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat/` | Full RAG answer (non-streaming) |
| `POST` | `/api/chat/stream/` | SSE streaming RAG answer |
| `GET` | `/api/sessions/` | List chat sessions |
| `POST` | `/api/sessions/` | Create new session |
| `GET` | `/api/sessions/<id>/` | Get session message history |
| `DELETE` | `/api/sessions/<id>/` | Delete session |

### Search & Debug

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/search/` | Semantic similarity search |
| `GET` | `/api/search/stats/` | ChromaDB collection stats |
| `GET` | `/api/debug/documents/` | All docs with chunk stats |
| `GET` | `/api/debug/documents/<id>/chunks/` | All chunks for a document |
| `POST` | `/api/debug/documents/<id>/rechunk/` | Re-run chunking with new settings |

### Chat stream request body
```json
{
  "question": "What is the patient's haemoglobin level?",
  "document_id": "uuid-string",
  "session_id": "uuid-string",
  "top_k": 5
}
```

### Chat stream SSE events
```
data: {"type": "token", "text": "The"}
data: {"type": "token", "text": " haemoglobin"}
...
data: {"type": "done", "sources": [...], "citations_used": ["[Source 1]"], "session_id": "..."}
data: {"type": "error", "message": "..."}
```

---

## Management Commands

```bash
# Chunk a specific document
python manage.py chunk_document <uuid>

# Chunk all pending documents
python manage.py chunk_document --all

# Force re-chunk with a specific strategy
python manage.py chunk_document <uuid> --strategy semantic --force

# Embed and index a document into ChromaDB
python manage.py index_document <uuid>

# Index all ready but unembedded documents
python manage.py index_document --all

# Check ChromaDB stats
python manage.py index_document --stats

# Bulk ingest a folder of PDFs
python manage.py ingest_pdfs /path/to/medical_docs/

# Dry run — preview what would be ingested
python manage.py ingest_pdfs /path/to/docs/ --dry-run

# Skip already-processed documents (idempotent)
python manage.py ingest_pdfs /path/to/docs/ --skip-existing
```

---

## Testing

### Key test scenarios before going to production:

**Upload & OCR**
- Upload a scanned PDF (image-only) — should auto-OCR and extract text
- Upload a 20.1 MB file — should reject with size error
- Upload a `.jpg` renamed to `.pdf` — magic bytes check should reject it
- Upload same file twice — both should be accepted as separate documents

**RAG Quality**
- Ask a question clearly answered in the document — citation should point to correct page
- Ask something not in the document — should say "not found" not hallucinate
- Ask a vague one-word query — should not crash

**Streaming**
- Check DevTools → Network → `/api/chat/stream/` shows `text/event-stream` content type
- Disconnect mid-stream — error message should appear in bubble, UI should not freeze

**Sessions**
- Ask 3 questions, close tab, reopen — history should load with citations
- Delete a session — should disappear from sidebar

**Chunk Inspector**
- Click "Embed + store in ChromaDB" — vector count should increase
- Run a search — results should appear with similarity scores
- Change chunk strategy and re-chunk — chunk count should update

---

## Roadmap

- [ ] User authentication (JWT)
- [ ] Multi-user support with data isolation
- [ ] Celery + Redis for async background processing
- [ ] Support for DOCX, TXT, CSV uploads
- [ ] Fine-tuned medical embedding model
- [ ] Docker + docker-compose setup
- [ ] Deploy to Railway / Render with one-click config

---

## Author

**Tinesh Ramrakhiyani**
- Email: tineshramrakhiyani@gmail.com
- LinkedIn: [linkedin.com/in/tinesh](https://linkedin.com/in/tinesh)
- GitHub: [github.com/tinesh](https://github.com/tinesh)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

> Built as a learning project to explore end-to-end RAG system design, vector databases, LLM integration, and production Django backend development.
