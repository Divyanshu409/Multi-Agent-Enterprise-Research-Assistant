from __future__ import annotations

import os

from src.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):

    def __init__(self, model: str, api_key: str | None = None):
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "openai package not installed. Run `pip install openai`."
            ) from e

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Export it or put it in .env "
                "(see .env.example), or set LLM_PROVIDER=mock for offline dev/tests."
            )
        self._client = openai.OpenAI(api_key=key)
        self._model = model

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    @property
    def name(self) -> str:
        return f"openai:{self._model}"
