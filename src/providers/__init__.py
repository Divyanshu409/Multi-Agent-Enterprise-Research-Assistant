from src.providers.base import EmbeddingProvider, LLMProvider
from src.providers.embeddings import get_embedding_provider


def get_llm_provider(provider_name: str, **kwargs) -> LLMProvider:

    if provider_name == "anthropic":
        from src.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=kwargs["model"], api_key=kwargs.get("api_key"))
    if provider_name == "openai":
        from src.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model=kwargs["model"], api_key=kwargs.get("api_key"))
    if provider_name == "gemini":
        from src.providers.gemini_provider import GeminiProvider

        return GeminiProvider(model=kwargs["model"], api_key=kwargs.get("api_key"))
    if provider_name == "mock":
        from src.providers.mock_provider import MockProvider

        return MockProvider()
    raise ValueError(f"Unknown LLM provider: {provider_name}")


__all__ = ["LLMProvider", "EmbeddingProvider", "get_llm_provider", "get_embedding_provider"]
