from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


def _register_and_login(client: TestClient) -> dict:
    """Helper: register and login, return auth headers."""
    email = f"{uuid4().hex}@example.com"
    resp = client.post(
        "/v1/auth/register",
        json={"email": email, "password": "password123"},
    )
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


SAMPLE_TEXT = b"Hello world, this is a short test document."

SAMPLE_MD = b"# Title\n\nThis is a **markdown** test document."


class TestDocumentUpload:
    def test_upload_text_file(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        resp = client.post(
            "/v1/documents",
            headers=headers,
            files={"file": ("hello.txt", SAMPLE_TEXT, "text/plain")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["filename"] == "hello.txt"
        assert data["content_type"] == "text/plain"
        assert data["id"].startswith("doc-")

    def test_upload_markdown(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        resp = client.post(
            "/v1/documents",
            headers=headers,
            files={"file": ("readme.md", SAMPLE_MD, "text/markdown")},
        )
        assert resp.status_code == 201
        assert resp.json()["content_type"] == "text/markdown"

    def test_upload_requires_auth(self) -> None:
        client = TestClient(app)
        resp = client.post(
            "/v1/documents",
            files={"file": ("test.txt", SAMPLE_TEXT, "text/plain")},
        )
        assert resp.status_code == 401


class TestDocumentList:
    def test_list_documents(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        # Upload two docs
        client.post(
            "/v1/documents",
            headers=headers,
            files={"file": ("a.txt", b"Content A", "text/plain")},
        )
        client.post(
            "/v1/documents",
            headers=headers,
            files={"file": ("b.txt", b"Content B", "text/plain")},
        )

        resp = client.get("/v1/documents", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 2
        filenames = {d["filename"] for d in data["documents"]}
        assert "a.txt" in filenames
        assert "b.txt" in filenames

    def test_list_documents_empty(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        resp = client.get("/v1/documents", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_list_requires_auth(self) -> None:
        client = TestClient(app)
        resp = client.get("/v1/documents")
        assert resp.status_code == 401


class TestDocumentGetDelete:
    def test_get_document(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        create_resp = client.post(
            "/v1/documents",
            headers=headers,
            files={"file": ("doc.txt", b"Some content", "text/plain")},
        )
        doc_id = create_resp.json()["id"]

        resp = client.get(f"/v1/documents/{doc_id}", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == doc_id
        assert resp.json()["filename"] == "doc.txt"

    def test_get_document_not_found(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        resp = client.get("/v1/documents/doc-nonexistent", headers=headers)
        assert resp.status_code == 404

    def test_get_other_users_document_returns_404(self) -> None:
        client = TestClient(app)
        headers_a = _register_and_login(client)
        headers_b = _register_and_login(client)

        create_resp = client.post(
            "/v1/documents",
            headers=headers_a,
            files={"file": ("secret.txt", b"Secret", "text/plain")},
        )
        doc_id = create_resp.json()["id"]

        resp = client.get(f"/v1/documents/{doc_id}", headers=headers_b)
        assert resp.status_code == 404

    def test_delete_document(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        create_resp = client.post(
            "/v1/documents",
            headers=headers,
            files={"file": ("delete-me.txt", b"Delete me", "text/plain")},
        )
        doc_id = create_resp.json()["id"]

        resp = client.delete(f"/v1/documents/{doc_id}", headers=headers)
        assert resp.status_code == 204

        # Verify gone
        get_resp = client.get(f"/v1/documents/{doc_id}", headers=headers)
        assert get_resp.status_code == 404

    def test_delete_document_not_found(self) -> None:
        client = TestClient(app)
        headers = _register_and_login(client)

        resp = client.delete("/v1/documents/doc-nonexistent", headers=headers)
        assert resp.status_code == 404

    def test_delete_requires_auth(self) -> None:
        client = TestClient(app)
        resp = client.delete("/v1/documents/doc-anything")
        assert resp.status_code == 401
