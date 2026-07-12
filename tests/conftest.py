from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_ollama() -> AsyncGenerator[MagicMock, None]:
    """Mock ChatService._call_ollama so tests don't need a running Ollama instance."""
    from app.services.chat_service import ChatService

    with patch.object(ChatService, "_call_ollama", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "id": "ollama-test",
            "model": "demo-chat",
            "created": 1234567890,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Mock Ollama reply: This is a test response.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        yield mock
