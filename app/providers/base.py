from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence

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
    # Raw tool calls emitted by the model (OpenAI-compatible shape), e.g.
    # ``[{"id": "call_...", "type": "function", "function": {...}}]``.
    # Empty when the model answered directly.
    tool_calls: list[dict] = []


class LLMStreamChunk(BaseModel):
    """A single chunk yielded during streaming.

    Every chunk carries ``content`` (the delta). The final chunk sets
    ``finish_reason`` — earlier chunks leave it ``None``.

    ``conversation_id`` is set on a leading meta chunk when a conversation
    was auto-created (the request had no ``conversation_id``) so the client
    learns the new conversation.
    """

    content: str = ""
    finish_reason: str | None = None
    conversation_id: str | None = None
    # Set when the model called a tool during the agentic loop — the client
    # can surface "calling tool X" while the tool runs.
    tool_call: str | None = None


class BaseLLMProvider(ABC):
    """Abstract interface every LLM provider must implement."""

    @abstractmethod
    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResult:
        """Non-streaming completion — returns the full result at once.

        ``tools`` is a list of OpenAI-compatible tool definitions
        (``{"type": "function", "function": {...}}``). When the model decides
        to call a tool, ``LLMResult.tool_calls`` is populated.
        """
        ...

    @abstractmethod
    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Streaming completion — yields chunks as they arrive.

        The last yielded chunk carries ``finish_reason``.
        """
        ...


class BaseEmbeddingProvider(ABC):
    """Abstract interface for text embedding providers."""

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Convert a sequence of texts into a list of embedding vectors.

        Args:
            texts: One or more text strings to embed.

        Returns:
            A list of vectors, one per input text. Each vector is a
            ``list[float]`` of fixed dimension (e.g. 768 for nomic-embed-text).

        Raises:
            ConnectionError: Provider unreachable.
            RuntimeError: Provider returned an error.
        """
        ...
