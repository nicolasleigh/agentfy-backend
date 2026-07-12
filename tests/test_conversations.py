from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


def _register_and_login(client: TestClient) -> dict:
    """Helper: register and login, return auth headers + user info."""
    email = f"{uuid4().hex}@example.com"
    client.post(
        "/v1/auth/register",
        json={"email": email, "password": "password123"},
    )
    login_resp = client.post(
        "/v1/auth/login",
        json={"email": email, "password": "password123"},
    )
    token = login_resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestConversations:
    def test_create_conversation(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        resp = client.post(
            "/v1/conversations",
            headers=headers,
            json={"title": "My Chat"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "My Chat"
        assert data["id"].startswith("conv-")
        assert "created_at" in data
        assert "updated_at" in data

    def test_create_conversation_default_title(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        resp = client.post(
            "/v1/conversations",
            headers=headers,
            json={},
        )
        assert resp.status_code == 201
        assert resp.json()["title"] == "New Chat"

    def test_create_conversation_requires_auth(self) -> None:
        client = TestClient(app)
        resp = client.post(
            "/v1/conversations",
            json={"title": "Hacker Attempt"},
        )
        assert resp.status_code == 401

    def test_list_conversations(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        # Create two conversations
        client.post("/v1/conversations", headers=headers, json={"title": "Chat A"})
        client.post("/v1/conversations", headers=headers, json={"title": "Chat B"})

        resp = client.get("/v1/conversations", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 2
        assert len(data["conversations"]) >= 2
        titles = {c["title"] for c in data["conversations"]}
        assert "Chat A" in titles
        assert "Chat B" in titles

    def test_list_conversations_empty(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        resp = client.get("/v1/conversations", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["conversations"] == []

    def test_list_conversations_requires_auth(self) -> None:
        client = TestClient(app)
        resp = client.get("/v1/conversations")
        assert resp.status_code == 401

    def test_get_conversation(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        create_resp = client.post(
            "/v1/conversations", headers=headers, json={"title": "My Chat"}
        )
        conv_id = create_resp.json()["id"]

        resp = client.get(f"/v1/conversations/{conv_id}", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == conv_id
        assert resp.json()["title"] == "My Chat"

    def test_get_conversation_not_found(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        resp = client.get(
            "/v1/conversations/conv-nonexistent", headers=headers
        )
        assert resp.status_code == 404

    def test_get_other_users_conversation_returns_404(self) -> None:
        client = TestClient(app)
        headers_a = _register_and_login(client)
        headers_b = _register_and_login(client)

        create_resp = client.post(
            "/v1/conversations", headers=headers_a, json={"title": "Secret Chat"}
        )
        conv_id = create_resp.json()["id"]

        # User B should not see User A's conversation
        resp = client.get(f"/v1/conversations/{conv_id}", headers=headers_b)
        assert resp.status_code == 404

    def test_update_conversation_title(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        create_resp = client.post(
            "/v1/conversations", headers=headers, json={"title": "Old Title"}
        )
        conv_id = create_resp.json()["id"]

        resp = client.patch(
            f"/v1/conversations/{conv_id}",
            headers=headers,
            json={"title": "New Title"},
        )
        assert resp.status_code == 200
        assert resp.json()["title"] == "New Title"

        # Verify persistence
        get_resp = client.get(f"/v1/conversations/{conv_id}", headers=headers)
        assert get_resp.json()["title"] == "New Title"

    def test_update_conversation_requires_auth(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        create_resp = client.post(
            "/v1/conversations", headers=headers, json={"title": "Title"}
        )
        conv_id = create_resp.json()["id"]

        resp = client.patch(
            f"/v1/conversations/{conv_id}",
            json={"title": "Hacked"},
        )
        assert resp.status_code == 401

    def test_delete_conversation(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        create_resp = client.post(
            "/v1/conversations", headers=headers, json={"title": "Delete Me"}
        )
        conv_id = create_resp.json()["id"]

        resp = client.delete(f"/v1/conversations/{conv_id}", headers=headers)
        assert resp.status_code == 204

        # Verify gone
        get_resp = client.get(f"/v1/conversations/{conv_id}", headers=headers)
        assert get_resp.status_code == 404

    def test_delete_conversation_not_found(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        resp = client.delete(
            "/v1/conversations/conv-nonexistent", headers=headers
        )
        assert resp.status_code == 404

    def test_delete_conversation_requires_auth(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        create_resp = client.post(
            "/v1/conversations", headers=headers, json={"title": "Title"}
        )
        conv_id = create_resp.json()["id"]

        resp = client.delete(f"/v1/conversations/{conv_id}")
        assert resp.status_code == 401


class TestChatCompletionWithConversation:
    def test_chat_auto_creates_conversation(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        chat_resp = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "demo-chat",
                "messages": [{"role": "user", "content": "Hello world"}],
            },
        )
        assert chat_resp.status_code == 200

        # A conversation should have been auto-created
        conv_resp = client.get("/v1/conversations", headers=headers)
        assert conv_resp.status_code == 200
        titles = [c["title"] for c in conv_resp.json()["conversations"]]
        assert "Hello world" in titles

    def test_chat_with_existing_conversation_id(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        # Create a conversation first
        conv_resp = client.post(
            "/v1/conversations", headers=headers, json={"title": "My Chat"}
        )
        conv_id = conv_resp.json()["id"]

        # Send a chat message referencing the conversation
        chat_resp = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "demo-chat",
                "messages": [{"role": "user", "content": "Keep chatting"}],
                "conversation_id": conv_id,
            },
        )
        assert chat_resp.status_code == 200
        assert chat_resp.json()["choices"][0]["message"]["content"] == (
            "Mock Ollama reply: This is a test response."
        )

    def test_chat_with_nonexistent_conversation_id(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        chat_resp = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "demo-chat",
                "messages": [{"role": "user", "content": "Hello"}],
                "conversation_id": "conv-nonexistent",
            },
        )
        assert chat_resp.status_code == 404

    def test_chat_with_other_users_conversation_id(self) -> None:
        client = TestClient(app)
        headers_a = _register_and_login(client)
        headers_b = _register_and_login(client)

        # User A creates a conversation
        conv_resp = client.post(
            "/v1/conversations", headers=headers_a, json={"title": "Secret"}
        )
        conv_id = conv_resp.json()["id"]

        # User B tries to use User A's conversation
        chat_resp = client.post(
            "/v1/chat/completions",
            headers=headers_b,
            json={
                "model": "demo-chat",
                "messages": [{"role": "user", "content": "Hello"}],
                "conversation_id": conv_id,
            },
        )
        assert chat_resp.status_code == 404


class TestConversationMessages:
    def test_get_messages_after_chat(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        # Send a chat — auto-creates a conversation
        chat_resp = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "demo-chat",
                "messages": [{"role": "user", "content": "Message one"}],
            },
        )
        assert chat_resp.status_code == 200

        # Find the auto-created conversation
        conv_resp = client.get("/v1/conversations", headers=headers)
        conv_id = conv_resp.json()["conversations"][0]["id"]

        # Get messages
        msg_resp = client.get(
            f"/v1/conversations/{conv_id}/messages", headers=headers
        )
        assert msg_resp.status_code == 200
        data = msg_resp.json()
        assert data["total"] == 2  # user + assistant
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Message one"
        assert data["messages"][1]["role"] == "assistant"
        assert data["messages"][1]["content"] == "Mock Ollama reply: This is a test response."

    def test_get_messages_multiple_turns(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        # Create a conversation explicitly
        conv_resp = client.post(
            "/v1/conversations", headers=headers, json={"title": "Multi-turn"}
        )
        conv_id = conv_resp.json()["id"]

        # Send two messages in the same conversation
        client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "demo-chat",
                "messages": [{"role": "user", "content": "First"}],
                "conversation_id": conv_id,
            },
        )
        client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "demo-chat",
                "messages": [{"role": "user", "content": "Second"}],
                "conversation_id": conv_id,
            },
        )

        msg_resp = client.get(
            f"/v1/conversations/{conv_id}/messages", headers=headers
        )
        assert msg_resp.status_code == 200
        data = msg_resp.json()
        assert data["total"] == 4  # user+assistant, user+assistant
        assert data["messages"][0]["content"] == "First"
        assert data["messages"][1]["role"] == "assistant"
        assert data["messages"][2]["content"] == "Second"
        assert data["messages"][3]["role"] == "assistant"

    def test_get_messages_empty_conversation(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        conv_resp = client.post(
            "/v1/conversations", headers=headers, json={"title": "Empty"}
        )
        conv_id = conv_resp.json()["id"]

        msg_resp = client.get(
            f"/v1/conversations/{conv_id}/messages", headers=headers
        )
        assert msg_resp.status_code == 200
        data = msg_resp.json()
        assert data["total"] == 0
        assert data["messages"] == []

    def test_get_messages_requires_auth(self) -> None:
        client = TestClient(app)
        resp = client.get("/v1/conversations/conv-xxx/messages")
        assert resp.status_code == 401

    def test_get_messages_conversation_not_found(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)
        resp = client.get(
            "/v1/conversations/conv-nonexistent/messages", headers=headers
        )
        assert resp.status_code == 404

    def test_get_messages_other_users_conversation(self) -> None:
        client = TestClient(app)
        headers_a = _register_and_login(client)
        headers_b = _register_and_login(client)

        conv_resp = client.post(
            "/v1/conversations", headers=headers_a, json={"title": "Secret"}
        )
        conv_id = conv_resp.json()["id"]

        resp = client.get(
            f"/v1/conversations/{conv_id}/messages", headers=headers_b
        )
        assert resp.status_code == 404
