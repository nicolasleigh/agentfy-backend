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
    assert chat_response.json()["choices"][0]["message"]["content"] == (
        "Demo backend received: hello"
    )


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
