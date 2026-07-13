from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from pydantic import BaseModel


class LLMResult(BaseModel):
    """Normalised result from any LLM provider (non-streaming)."""

    content: str
    role: str = "assistant"
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str
    created: int  # unix timestamp


class LLMStreamChunk(BaseModel):
    """A single chunk yielded during streaming.

    Every chunk carries ``content`` (the delta). The final chunk sets
    ``finish_reason`` — earlier chunks leave it ``None``.
    """

    content: str = ""
    finish_reason: str | None = None


class BaseLLMProvider(ABC):
    """Abstract interface every LLM provider must implement."""

    @abstractmethod
    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
    ) -> LLMResult:
        """Non-streaming completion — returns the full result at once."""
        ...

    @abstractmethod
    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Streaming completion — yields chunks as they arrive.

        The last yielded chunk carries ``finish_reason``.
        """
        ...
        # yield  # pragma: no cover — makes the method an async generator
