"""
MediAssit/gemini.py
────────────────
Initialises the modern Google GenAI SDK and provides two lightweight
wrappers used throughout the project:

    GeminiService.chat(prompt)        → plain text reply
    GeminiService.embed(text)         → 768-dim float list
    GeminiService.health_check()      → (ok: bool, message: str)

Run this file directly to test your API key:
    python MediAssit/gemini.py
"""

import os
import sys
import textwrap

from google import genai
from google.genai import types
from django.conf import settings

print("Script started")

# ── SDK initialisation ────────────────────────────────────────────────────────


def _get_api_key():
    api_key = os.environ.get("GEMINI_API_KEY") or getattr(
        settings, "GEMINI_API_KEY", ""
    )
    print("API KEY FOUND:", bool(api_key))
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set.\n"
            "Add it to your .env file:  GEMINI_API_KEY=your_key_here"
        )
    return api_key


# ── Models ────────────────────────────────────────────────────────────────────

CHAT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"


# ── Service class ─────────────────────────────────────────────────────────────


class GeminiService:
    """
    Thin wrapper around the modern Gemini SDK.
    All methods are static/class methods — no instance state needed.
    """

    _client = None

    @classmethod
    def _ensure_init(cls):
        # The new SDK uses a Client object instead of a global configuration
        if cls._client is None:
            api_key = _get_api_key()
            cls._client = genai.Client(api_key=api_key)

    # ── Text generation ───────────────────────────────────────────

    @classmethod
    def chat(cls, prompt: str, system_prompt: str = "") -> str:
        """
        Send a single prompt to Gemini and return the text response.
        """
        cls._ensure_init()
        print("Calling Gemini API...")

        # We maintain your string concatenation approach for perfect backward compatibility
        full_prompt = f"{system_prompt}\n\nUser: {prompt}" if system_prompt else prompt

        response = cls._client.models.generate_content(
            model=CHAT_MODEL,
            contents=full_prompt
        )
        return response.text

    # ── Streaming generation (used in Phase 5 chat endpoint) ─────

    @classmethod
    def stream(cls, prompt: str, system_prompt: str = ""):
        """
        Yields text chunks as Gemini streams the response.
        Use this in your Django StreamingHttpResponse view.
        """
        cls._ensure_init()
        full_prompt = f"{system_prompt}\n\nUser: {prompt}" if system_prompt else prompt

        # The new SDK has a dedicated method for streaming
        response = cls._client.models.generate_content_stream(
            model=CHAT_MODEL,
            contents=full_prompt
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    # ── Embedding ─────────────────────────────────────────────────

    @classmethod
    def embed(cls, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        """
        Convert text into a 768-dimensional embedding vector.
        """
        cls._ensure_init()

        # The new SDK uses EmbedContentConfig for parameters and requires uppercase task types
        result = cls._client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type.upper(), 
                output_dimensionality=768
            )
        )
        # Navigating the modern response object to get the float list
        return result.embeddings[0].values

    # ── Health check ──────────────────────────────────────────────

    @classmethod
    def health_check(cls) -> tuple[bool, str]:
        """
        Sends a tiny prompt and verifies the API responds.
        Returns (True, success_message) or (False, error_message).
        """
        try:
            response = cls.chat(
                prompt="Reply with exactly three words: Gemini is ready.",
                system_prompt="You are a test assistant. Follow instructions exactly.",
            )
            return True, f'Gemini responded: "{response.strip()}"'
        except EnvironmentError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Gemini API error: {e}"


# ── Standalone hello-world test ───────────────────────────────────────────────
# Run:  python MediAssit/gemini.py

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 55)
    print("  MediAssist — Gemini API Health Check (Modern SDK)")
    print("=" * 55)

    # 1. Basic chat test
    print("\n[1] Chat test ...")
    ok, message = GeminiService.health_check()
    status_icon = "OK" if ok else "FAIL"
    print(f"    [{status_icon}] {message}")

    # 2. Embedding test
    if ok:
        print("\n[2] Embedding test ...")
        try:
            vec = GeminiService.embed(
                "What are the symptoms of type 2 diabetes?",
                task_type="RETRIEVAL_QUERY",
            )
            print(f"    [OK] Vector generated — dimensions: {len(vec)}")
            print(f"         First 5 values: {[round(v, 6) for v in vec[:5]]}")
        except Exception as e:
            print(f"    [FAIL] {e}")

    # 3. Healthcare system prompt test
    if ok:
        print("\n[3] System prompt test (healthcare context) ...")
        try:
            answer = GeminiService.chat(
                prompt="What is hypertension in one sentence?",
                system_prompt=(
                    "You are MediAssist, a healthcare AI assistant. "
                    "Answer medical questions clearly and concisely. "
                    "Always remind users to consult a doctor."
                ),
            )
            wrapped = textwrap.fill(
                answer.strip(), width=50, initial_indent="        "
            )
            print(f"    [OK] Response:\n{wrapped}")
        except Exception as e:
            print(f"    [FAIL] {e}")

    print("\n" + "=" * 55)
    print("  All checks complete.")
    print("=" * 55)
    sys.exit(0 if ok else 1)