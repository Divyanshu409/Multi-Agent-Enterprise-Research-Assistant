from __future__ import annotations

import os

import requests

from src.providers.base import LLMProvider


class GeminiProvider(LLMProvider):
    """Wraps Google's Gemini API via plain HTTP requests instead of the
    google-genai SDK. The SDK currently mishandles the newer 'AQ.' prefixed
    API keys (401 ACCESS_TOKEN_TYPE_UNSUPPORTED) that Google started issuing
    from AI Studio in mid-2026 -- calling the REST endpoint directly with the
    key in the x-goog-api-key header works fine, confirmed via curl."""

    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model: str, api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey "
                "and put it in .env, or set LLM_PROVIDER=mock for offline dev/tests."
            )
        self._key = key
        self._model = model

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        import time

        url = f"{self._BASE_URL}/{self._model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._key}
        payload = {
            "contents": [{"parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

        last_error = None
        for attempt in range(3):
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            if response.status_code == 503:
                wait = 10 * (attempt + 1)
                print(f"  [gemini] server busy (503), retrying in {wait}s...")
                time.sleep(wait)
                last_error = response
                continue
            response.raise_for_status()
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return self._strip_markdown_fences(text)

        last_error.raise_for_status()

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    @property
    def name(self) -> str:
        return f"gemini:{self._model}"