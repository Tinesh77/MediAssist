"""
Add these to your existing api/urls.py
"""
from django.urls import path
from .debug_views import DebugDocumentListView, DebugChunkListView, DebugRechunkView

# Append to your existing urlpatterns list:
debug_urlpatterns = [
    path("debug/documents/",                       DebugDocumentListView.as_view(), name="debug-doc-list"),
    path("debug/documents/<uuid:doc_id>/chunks/",  DebugChunkListView.as_view(),    name="debug-chunk-list"),
    path("debug/documents/<uuid:doc_id>/rechunk/", DebugRechunkView.as_view(),      name="debug-rechunk"),
]