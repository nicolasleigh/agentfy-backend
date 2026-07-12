from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


def test_register_login_me_and_authenticated_chat_completion() -> None:
    client = TestClient(app)
    email = f"{uuid4().hex}@example.com"
    password = "password123"

    register_response = client.post(
        "/v1/auth/register",
        json={"email": email, "password": password, "name": "Alice"},
    )
    assert register_response.status_code == 201
    register_body = register_response.json()
    assert register_body["token_type"] == "bearer"
    assert register_body["access_token"]
    assert register_body["user"]["email"] == email

    login_response = client.post(
        "/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]

    me_response = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_response.status_code == 200
    assert me_response.json()["email"] == email

    chat_response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": "demo-chat",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert chat_response.status_code == 200
    data = chat_response.json()
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == (
        "Mock Ollama reply: This is a test response."
    )
    assert data["usage"]["total_tokens"] == 15


def test_chat_completion_requires_authentication() -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "demo-chat",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 401


def test_chat_completion_llm_error() -> None:
    """Verify that an LLM provider error is surfaced as a proper HTTP error."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)

    # Register and login
    from uuid import uuid4
    email = f"{uuid4().hex}@example.com"
    resp = client.post(
        "/v1/auth/register",
        json={"email": email, "password": "password123"},
    )
    assert resp.status_code == 201
    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    mock_provider = MagicMock()
    mock_provider.chat_completion = AsyncMock(
        side_effect=ConnectionError("Cannot connect to Ollama")
    )

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_provider):
        chat_resp = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "demo-chat",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert chat_resp.status_code == 503
        assert "Cannot connect" in chat_resp.json()["detail"]
