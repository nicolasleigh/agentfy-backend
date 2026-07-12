from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.providers.base import LLMResult


@pytest.fixture(autouse=True)
def mock_llm_provider() -> Generator[MagicMock, None, None]:
    """Mock the LLM provider so tests never make real HTTP calls.

    Patches ``get_llm_provider`` to return a mock whose ``chat_completion``
    method returns a controlled ``LLMResult``.
    """
    mock_provider = MagicMock()
    mock_provider.chat_completion = AsyncMock(return_value=LLMResult(
        content="Mock Ollama reply: This is a test response.",
        model="demo-chat",
        created=1234567890,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    ))

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_provider):
        yield mock_provider
