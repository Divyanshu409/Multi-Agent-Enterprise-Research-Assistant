from __future__ import annotations

import os

from src.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):

    def __init__(self, model: str, api_key: str | None = None):
        try:
            import anthropic
        except ImportError as e: 
            raise RuntimeError(
                "anthropic package not installed. Run `pip install anthropic`."
            ) from e

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it or put it in .env "
                "(see .env.example), or set LLM_PROVIDER=mock for offline dev/tests."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        
        return "".join(block.text for block in response.content if block.type == "text")

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"
