from functools import lru_cache

from app.core.config import settings
from app.providers.base import BaseLLMProvider, LLMResult
from app.providers.ollama import OllamaProvider
from app.providers.openai import OpenAIProvider

__all__ = [
    "BaseLLMProvider",
    "LLMResult",
    "get_llm_provider",
    "OllamaProvider",
    "OpenAIProvider",
]

_PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
}


@lru_cache
def get_llm_provider() -> BaseLLMProvider:
    """Return a singleton provider instance based on ``settings.llm_provider``."""
    name = settings.llm_provider.lower()
    cls = _PROVIDER_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(_PROVIDER_REGISTRY)
        raise ValueError(
            f"Unknown LLM provider: '{name}'. Available: {available}"
        )
    return cls()
