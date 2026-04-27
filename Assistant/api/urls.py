"""
api/urls.py  — complete URL config for all phases
──────────────────────────────────────────────────
Replace your existing api/urls.py with this file.
"""
from django.urls import path

from .views import DocumentDetailView, DocumentListView, DocumentUploadView
from .debug_urls import debug_urlpatterns
from .retrieval_views import IndexDocumentView, SemanticSearchView, SearchStatsView
from .chat_views import (
    ChatView,
    ChatStreamView,
    SessionListView,
    SessionDetailView,
)

urlpatterns = [
    # ── Phase 1: Document CRUD ────────────────────────────────────
    path("documents/",
         DocumentListView.as_view(),   name="document-list"),
    path("documents/upload/",
         DocumentUploadView.as_view(), name="document-upload"),
    path("documents/<uuid:doc_id>/",
         DocumentDetailView.as_view(), name="document-detail"),

    # ── Phase 3: Indexing + semantic search ───────────────────────
    path("documents/<uuid:doc_id>/index/",
         IndexDocumentView.as_view(),  name="document-index"),
    path("search/",
         SemanticSearchView.as_view(), name="semantic-search"),
    path("search/stats/",
         SearchStatsView.as_view(),    name="search-stats"),

    # ── Phase 4: RAG chat ─────────────────────────────────────────
    path("chat/",
         ChatView.as_view(),           name="chat"),
    path("chat/stream/",
         ChatStreamView.as_view(),     name="chat-stream"),

    # ── Phase 4: Session management ───────────────────────────────
    path("sessions/",
         SessionListView.as_view(),    name="session-list"),
    path("sessions/<uuid:session_id>/",
         SessionDetailView.as_view(),  name="session-detail"),
]

urlpatterns += debug_urlpatterns