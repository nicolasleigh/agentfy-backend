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


def test_chat_completion_ollama_error() -> None:
    """Verify that an Ollama error is surfaced as a proper HTTP error."""
    from unittest.mock import AsyncMock, patch

    from fastapi.testclient import TestClient

    from app.main import app
    from app.services.chat_service import ChatService

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

    with patch.object(ChatService, "_call_ollama", new_callable=AsyncMock) as mock:
        from fastapi import HTTPException, status
        mock.side_effect = HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cannot connect to Ollama",
        )

        chat_resp = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "demo-chat",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert chat_resp.status_code == 503
