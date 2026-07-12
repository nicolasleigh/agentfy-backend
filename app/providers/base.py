from abc import ABC, abstractmethod

from pydantic import BaseModel


class LLMResult(BaseModel):
    """Normalised result from any LLM provider."""

    content: str
    role: str = "assistant"
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str
    created: int  # unix timestamp


class BaseLLMProvider(ABC):
    """Abstract interface every LLM provider must implement."""

    @abstractmethod
    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
        stream: bool = False,
    ) -> LLMResult:
        """Send a chat completion request and return a normalised result.

        Args:
            model: Model name (e.g. ``"llama3.2"``, ``"gpt-4o"``).
            messages: List of ``{"role": …, "content": …}`` dicts.
            temperature: Sampling temperature (provider default if ``None``).
            stream: Whether to stream — providers may raise if unsupported.

        Returns:
            LLMResult: Normalised response.

        Raises:
            ConnectionError: Provider unreachable.
            TimeoutError: Request timed out.
            RuntimeError: Provider returned an error (with detail in args).
        """
        ...
