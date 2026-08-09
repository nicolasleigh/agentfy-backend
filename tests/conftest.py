"""Test configuration.

Tests use SQLite (not PostgreSQL) because ``asyncpg`` does not support
the synchronous ``TestClient`` transport.  The ``DATABASE_URL`` env var
is set **before** any ``app`` import so that ``app.core.config.settings``
picks it up.
"""

import os

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./data/test.db"
# Known token for the MCP endpoint so tests can authenticate deterministically.
os.environ["MCP_AUTH_TOKEN"] = "test-mcp-token"


from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.providers.base import LLMResult, LLMStreamChunk


@pytest.fixture(autouse=True)
def mock_llm_provider() -> Generator[MagicMock, None, None]:
    """Mock the LLM provider so tests never make real HTTP calls."""
    mock_provider = MagicMock()

    # Non-streaming
    mock_provider.chat_completion = AsyncMock(return_value=LLMResult(
        content="Mock Ollama reply: This is a test response.",
        model="demo-chat",
        created=1234567890,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    ))

    # Streaming — async generator
    async def _stream(model, messages, temperature=None) -> AsyncGenerator[LLMStreamChunk, None]:
        yield LLMStreamChunk(content="Mock ")
        yield LLMStreamChunk(content="stream reply.")
        yield LLMStreamChunk(content="", finish_reason="stop")

    mock_provider.chat_completion_stream = _stream

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_provider):
        yield mock_provider


@pytest.fixture(autouse=True)
def mock_embedding_provider() -> Generator[MagicMock, None, None]:
    """Mock the embedding provider so tests never call the real Ollama."""
    mock_embedder = MagicMock()

    async def _embed(texts: list[str]) -> list[list[float]]:
        return [[0.042] * 768 for _ in texts]

    mock_embedder.embed = AsyncMock(side_effect=_embed)

    with patch("app.services.embedding_service.get_embedding_provider",
               return_value=mock_embedder):
        yield mock_embedder


@pytest.fixture(autouse=True)
def mock_rag_retrieve() -> Generator[MagicMock, None, None]:
    """Mock ``EmbeddingService.retrieve`` so RAG never hits pgvector SQL.

    By default returns an empty list (no context injected).
    RAG-specific tests override this fixture's return value.
    """
    from app.services.embedding_service import EmbeddingService

    mock_retrieve = AsyncMock(return_value=[])
    with patch.object(EmbeddingService, "retrieve", mock_retrieve):
        yield mock_retrieve
