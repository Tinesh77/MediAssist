"""
api/chat_views.py
──────────────────────────────────────────────────────────────────
Phase 4 API endpoints — chat with RAG-powered Gemini answers.

Endpoints:
    POST /api/chat/                      — full answer (non-streaming)
    POST /api/chat/stream/               — token-by-token SSE stream
    GET  /api/sessions/                  — list chat sessions for current user
    POST /api/sessions/                  — create a new session
    GET  /api/sessions/<id>/messages/    — message history for a session
    DELETE /api/sessions/<id>/           — delete a session

Request body for /api/chat/ and /api/chat/stream/:
    {
        "question":    "What are the symptoms of hypertension?",   required
        "session_id":  "uuid",     optional — uses/creates a session
        "document_id": "uuid",     optional — scope retrieval to one doc
        "document_ids": ["uuid"],  optional — scope to multiple docs
        "top_k":       5           optional — chunks to inject (default 5)
    }

Response from /api/chat/:
    {
        "success":        true,
        "answer":         "Hypertension symptoms include... [Source 1]",
        "citations_used": ["[Source 1]", "[Source 2]"],
        "tokens_used":    {"prompt": 1240, "completion": 380, "total": 1620},
        "sources": [
            {
                "n":               1,
                "citation_label":  "[Source 1]",
                "text":            "...",
                "page_start":      3,
                "page_end":        3,
                "page_label":      "p.3",
                "doc_title":       "JNC 8 Guidelines",
                "doc_id":          "uuid",
                "retrieval_score": 0.91,
                "rerank_score":    0.93
            }
        ],
        "session_id": "uuid"
    }
"""

import json
import logging

from django.http import JsonResponse, StreamingHttpResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def _get_or_create_session(session_id, user, document_ids=None):
    """
    Return an existing ChatSession or create a fresh one.
    Optionally scopes the session to specific documents.
    """
    from ..models import ChatSession, Document

    if session_id:
        try:
            qs = ChatSession.objects.all()
            if user and user.is_authenticated:
                qs = qs.filter(user=user)
            return qs.get(id=session_id)
        except ChatSession.DoesNotExist:
            pass   # fall through to create

    session = ChatSession.objects.create(
        user     = user if user and user.is_authenticated else None,
        is_active = True,
    )

    if document_ids:
        docs = Document.objects.filter(id__in=document_ids)
        session.documents.set(docs)

    return session


def _parse_body(request) -> dict:
    try:
        return json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return {}


def _resolve_document(document_id: str, document_ids: list):
    """
    Return (document, document_ids_list) ready for ChatService.
    Validates that the requested documents exist.
    """
    from ..models import Document

    document = None
    if document_id:
        try:
            document = Document.objects.get(id=document_id)
        except Document.DoesNotExist:
            return None, None, f"Document '{document_id}' not found."

    return document, document_ids or None, None


# ─────────────────────────────────────────────────────────────────
#  POST /api/chat/   — full non-streaming answer
# ─────────────────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name="dispatch")
class ChatView(View):

    def post(self, request):
        from ..services.prompt_builder import ChatService

        body         = _parse_body(request)
        question     = body.get("question", "").strip()
        session_id   = body.get("session_id")
        document_id  = body.get("document_id")
        document_ids = body.get("document_ids", [])
        top_k        = min(int(body.get("top_k", 5)), 10)

        if not question:
            return JsonResponse(
                {"success": False, "error": "'question' field is required."},
                status=400,
            )

        # Resolve document scope
        document, doc_ids, err = _resolve_document(document_id, document_ids)
        if err:
            return JsonResponse({"success": False, "error": err}, status=404)

        # Get / create session
        session = _get_or_create_session(
            session_id   = session_id,
            user         = request.user,
            document_ids = document_ids,
        )

        user_id = request.user.id if request.user.is_authenticated else None

        try:
            response = ChatService.answer(
                question       = question,
                session        = session,
                document       = document,
                document_ids   = doc_ids,
                user_id        = user_id,
                top_k_rerank   = top_k,
            )
        except Exception as exc:
            logger.error(f"[ChatView] Error: {exc}", exc_info=True)
            return JsonResponse(
                {"success": False, "error": f"Chat failed: {exc}"},
                status=500,
            )

        return JsonResponse({
            "success":        True,
            "answer":         response.answer,
            "citations_used": response.citations_used,
            "tokens_used":    response.tokens_used,
            "sources":        response.sources,
            "session_id":     str(session.id),
        })


# ─────────────────────────────────────────────────────────────────
#  POST /api/chat/stream/   — Server-Sent Events streaming
# ─────────────────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name="dispatch")
class ChatStreamView(View):
    """
    Streams Gemini's answer token-by-token using Server-Sent Events (SSE).

    SSE event format:
        data: {"type": "token", "text": "Hypertension"}
        data: {"type": "token", "text": " is"}
        ...
        data: {"type": "done", "sources": [...], "citations_used": [...], "session_id": "..."}

    Frontend usage:
        const es = new EventSource('/api/chat/stream/', ...);
        es.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            if (msg.type === 'token') appendText(msg.text);
            if (msg.type === 'done')  showSources(msg.sources);
        };
    """

    def post(self, request):
        from ..services.prompt_builder import ChatService

        body         = _parse_body(request)
        question     = body.get("question", "").strip()
        session_id   = body.get("session_id")
        document_id  = body.get("document_id")
        document_ids = body.get("document_ids", [])
        top_k        = min(int(body.get("top_k", 5)), 10)

        if not question:
            return JsonResponse(
                {"success": False, "error": "'question' is required."},
                status=400,
            )

        document, doc_ids, err = _resolve_document(document_id, document_ids)
        if err:
            return JsonResponse({"success": False, "error": err}, status=404)

        session = _get_or_create_session(
            session_id   = session_id,
            user         = request.user,
            document_ids = document_ids,
        )
        user_id = request.user.id if request.user.is_authenticated else None

        def event_stream():
            try:
                gen = ChatService.stream(
                    question       = question,
                    session        = session,
                    document       = document,
                    document_ids   = doc_ids,
                    user_id        = user_id,
                    top_k_rerank   = top_k,
                )

                # Yield tokens
                final_response = None
                try:
                    while True:
                        token = next(gen)
                        payload = json.dumps({"type": "token", "text": token})
                        yield f"data: {payload}\n\n"
                except StopIteration as e:
                    final_response = e.value

                # Send done event with sources
                if final_response:
                    done_payload = json.dumps({
                        "type":           "done",
                        "sources":        final_response.sources,
                        "citations_used": final_response.citations_used,
                        "session_id":     str(session.id),
                    })
                    yield f"data: {done_payload}\n\n"

            except Exception as exc:
                logger.error(f"[ChatStreamView] Stream error: {exc}", exc_info=True)
                error_payload = json.dumps({"type": "error", "message": str(exc)})
                yield f"data: {error_payload}\n\n"

        return StreamingHttpResponse(
            event_stream(),
            content_type = "text/event-stream",
            headers      = {
                "Cache-Control":  "no-cache",
                "X-Accel-Buffering": "no",  # disables nginx buffering
            },
        )


# ─────────────────────────────────────────────────────────────────
#  Session management endpoints
# ─────────────────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name="dispatch")
class SessionListView(View):
    """
    GET  /api/sessions/   — list sessions for current user
    POST /api/sessions/   — create a new session
    """

    def get(self, request):
        from ..models import ChatSession

        # Return all active sessions. When auth is added later,
        # uncomment the filter below to scope to the logged-in user.
        qs = ChatSession.objects.filter(is_active=True).order_by("-updated_at")
        # if request.user.is_authenticated:
        #     qs = qs.filter(user=request.user)

        sessions = []
        for s in qs[:50]:
            sessions.append({
                "id":            str(s.id),
                "title":         s.title or "Untitled",
                "message_count": s.message_count,
                "llm_model":     s.llm_model,
                "created_at":    s.created_at.isoformat(),
                "updated_at":    s.updated_at.isoformat(),
                "documents": [
                    {"id": str(d.id), "title": d.title}
                    for d in s.documents.all()
                ],
            })

        return JsonResponse({"success": True, "sessions": sessions})

    def post(self, request):
        from ..models import ChatSession

        session = ChatSession.objects.create(
            user      = request.user if request.user.is_authenticated else None,
            is_active = True,
        )
        return JsonResponse({
            "success":    True,
            "session_id": str(session.id),
        }, status=201)


@method_decorator(csrf_exempt, name="dispatch")
class SessionDetailView(View):
    """
    GET    /api/sessions/<id>/messages/   — full message history
    DELETE /api/sessions/<id>/            — soft-delete session
    """

    def _get_session(self, session_id, request):
        from ..models import ChatSession
        try:
            # No user filter for anonymous dev use.
            return ChatSession.objects.get(id=session_id)
        except ChatSession.DoesNotExist:
            return None

    def get(self, request, session_id):
        session = self._get_session(session_id, request)
        if not session:
            return JsonResponse({"success": False, "error": "Session not found."}, status=404)

        messages = []
        for msg in session.messages.order_by("created_at"):
            entry = {
                "id":         str(msg.id),
                "role":       msg.role,
                "content":    msg.content,
                "created_at": msg.created_at.isoformat(),
            }
            # Include sources for assistant messages
            if msg.is_assistant:
                entry["sources"] = [
                    {
                        "id":         str(c.id),
                        "text":       c.text[:200],
                        "page_start": c.page_start,
                        "page_end":   c.page_end,
                        "doc_title":  c.document.title,
                    }
                    for c in msg.source_chunks.select_related("document").all()
                ]
                entry["tokens"] = {
                    "prompt":     msg.prompt_tokens,
                    "completion": msg.completion_tokens,
                    "total":      msg.total_tokens,
                }
            messages.append(entry)

        return JsonResponse({
            "success":  True,
            "session":  {
                "id":      str(session.id),
                "title":   session.title or "Untitled",
                "created": session.created_at.isoformat(),
            },
            "messages": messages,
        })

    def delete(self, request, session_id):
        session = self._get_session(session_id, request)
        if not session:
            return JsonResponse({"success": False, "error": "Session not found."}, status=404)

        session.is_active = False
        session.save(update_fields=["is_active"])
        return JsonResponse({"success": True, "message": "Session deleted."})