"""Test configuration.

Tests use SQLite (not PostgreSQL) because ``asyncpg`` does not support
the synchronous ``TestClient`` transport.  The ``DATABASE_URL`` env var
is set **before** any ``app`` import so that ``app.core.config.settings``
picks it up.
"""

import os

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./data/test.db"


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
    async def _stream(**kwargs) -> AsyncGenerator[LLMStreamChunk, None]:
        yield LLMStreamChunk(content="Mock ")
        yield LLMStreamChunk(content="stream reply.")
        yield LLMStreamChunk(content="", finish_reason="stop")

    mock_provider.chat_completion_stream = _stream

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_provider):
        yield mock_provider
