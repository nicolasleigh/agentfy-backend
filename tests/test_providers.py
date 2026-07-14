"""Tests for the LLM / embedding provider abstraction layer."""

import pytest

from app.providers import (
    _EMBEDDING_PROVIDER_REGISTRY,
    _LLM_PROVIDER_REGISTRY,
    get_embedding_provider,
    get_llm_provider,
    OllamaEmbeddingProvider,
    OllamaProvider,
    OpenAIProvider,
)
from app.providers.base import LLMResult


# ---------------------------------------------------------------------------
# LLM Provider Registry
# ---------------------------------------------------------------------------


class TestLLMProviderRegistry:
    def test_llm_registry_contains_ollama(self) -> None:
        assert "ollama" in _LLM_PROVIDER_REGISTRY
        assert _LLM_PROVIDER_REGISTRY["ollama"] is OllamaProvider

    def test_llm_registry_contains_openai(self) -> None:
        assert "openai" in _LLM_PROVIDER_REGISTRY
        assert _LLM_PROVIDER_REGISTRY["openai"] is OpenAIProvider

    def test_default_llm_provider_is_ollama(self) -> None:
        provider = get_llm_provider()
        assert isinstance(provider, OllamaProvider)

    def test_llm_provider_is_singleton(self) -> None:
        assert get_llm_provider() is get_llm_provider()

    def test_unknown_llm_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            name = "nonexistent"
            cls = _LLM_PROVIDER_REGISTRY.get(name)
            if cls is None:
                available = ", ".join(_LLM_PROVIDER_REGISTRY)
                raise ValueError(
                    f"Unknown LLM provider: '{name}'. Available: {available}"
                )


# ---------------------------------------------------------------------------
# Embedding Provider Registry
# ---------------------------------------------------------------------------


class TestEmbeddingProviderRegistry:
    def test_embedding_registry_contains_ollama(self) -> None:
        assert "ollama" in _EMBEDDING_PROVIDER_REGISTRY
        assert _EMBEDDING_PROVIDER_REGISTRY["ollama"] is OllamaEmbeddingProvider

    def test_default_embedding_provider_is_ollama(self) -> None:
        provider = get_embedding_provider()
        assert isinstance(provider, OllamaEmbeddingProvider)

    def test_embedding_provider_is_singleton(self) -> None:
        assert get_embedding_provider() is get_embedding_provider()


# ---------------------------------------------------------------------------
# LLMResult model
# ---------------------------------------------------------------------------


class TestLLMResult:
    def test_default_values(self) -> None:
        result = LLMResult(
            content="Hello",
            model="test-model",
            created=1234567890,
        )
        assert result.content == "Hello"
        assert result.role == "assistant"
        assert result.finish_reason == "stop"
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0

    def test_custom_values(self) -> None:
        result = LLMResult(
            content="Hi",
            model="gpt-4o",
            created=100,
            role="assistant",
            finish_reason="length",
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
        )
        assert result.content == "Hi"
        assert result.model == "gpt-4o"
        assert result.finish_reason == "length"
        assert result.total_tokens == 60
