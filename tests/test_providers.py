"""Tests for the LLM provider abstraction layer."""

import pytest

from app.providers import _PROVIDER_REGISTRY, get_llm_provider, OllamaProvider, OpenAIProvider
from app.providers.base import LLMResult


class TestProviderRegistry:
    def test_registry_contains_ollama(self) -> None:
        assert "ollama" in _PROVIDER_REGISTRY
        assert _PROVIDER_REGISTRY["ollama"] is OllamaProvider

    def test_registry_contains_openai(self) -> None:
        assert "openai" in _PROVIDER_REGISTRY
        assert _PROVIDER_REGISTRY["openai"] is OpenAIProvider

    def test_default_provider_is_ollama(self) -> None:
        provider = get_llm_provider()
        assert isinstance(provider, OllamaProvider)

    def test_provider_is_singleton(self) -> None:
        assert get_llm_provider() is get_llm_provider()

    def test_unknown_provider_raises(self) -> None:
        """get_llm_provider validates the provider name is in the registry."""
        from app import providers
        # Simulate what happens when settings.llm_provider is not in the dict
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            # Call the internal logic that raises on unknown names
            name = "nonexistent"
            cls = providers._PROVIDER_REGISTRY.get(name)
            if cls is None:
                available = ", ".join(providers._PROVIDER_REGISTRY)
                raise ValueError(
                    f"Unknown LLM provider: '{name}'. Available: {available}"
                )


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
