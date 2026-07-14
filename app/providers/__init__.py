from functools import lru_cache

from app.core.config import settings
from app.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    LLMResult,
    LLMStreamChunk,
)
from app.providers.ollama import OllamaProvider
from app.providers.ollama_embedding import OllamaEmbeddingProvider
from app.providers.openai import OpenAIProvider

__all__ = [
    "BaseEmbeddingProvider",
    "BaseLLMProvider",
    "LLMResult",
    "LLMStreamChunk",
    "get_llm_provider",
    "get_embedding_provider",
    "OllamaProvider",
    "OllamaEmbeddingProvider",
    "OpenAIProvider",
]

# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------

_LLM_PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
}


@lru_cache
def get_llm_provider() -> BaseLLMProvider:
    """Return a singleton LLM provider instance based on ``settings.llm_provider``."""
    name = settings.llm_provider.lower()
    cls = _LLM_PROVIDER_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(_LLM_PROVIDER_REGISTRY)
        raise ValueError(
            f"Unknown LLM provider: '{name}'. Available: {available}"
        )
    return cls()


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------

_EMBEDDING_PROVIDER_REGISTRY: dict[str, type[BaseEmbeddingProvider]] = {
    "ollama": OllamaEmbeddingProvider,
}


@lru_cache
def get_embedding_provider() -> BaseEmbeddingProvider:
    """Return a singleton embedding provider instance.

    Currently always returns ``OllamaEmbeddingProvider``.
    A future ``EMBEDDING_PROVIDER`` setting can be added when other
    backends (OpenAI, HuggingFace, etc.) are implemented.
    """
    name = settings.llm_provider.lower()
    cls = _EMBEDDING_PROVIDER_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(_EMBEDDING_PROVIDER_REGISTRY)
        raise ValueError(
            f"No embedding provider for LLM provider '{name}'. "
            f"Available: {available}"
        )
    return cls()
