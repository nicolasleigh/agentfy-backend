"""Tests for the MCP server (Direction B: knowledge base search)."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.mcp.tools import list_documents, search_knowledge_base

TOKEN = "test-mcp-token"
ACCEPT = "application/json, text/event-stream"

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "1.0"},
    },
}


def _post(client: TestClient, body: dict, *, token: str | None = TOKEN) -> object:
    headers = {"Accept": ACCEPT}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return client.post("/mcp", json=body, headers=headers)


def _initialize(client: TestClient) -> str:
    """Initialize an MCP session and return its session id."""
    response = _post(client, INITIALIZE)
    assert response.status_code == 200
    session_id = response.headers.get("mcp-session-id")
    assert session_id, "initialize did not return a session id"
    return session_id


# ----------------------------------------------------------------------
# HTTP layer: auth + tool listing
# ----------------------------------------------------------------------


def test_mcp_rejects_requests_without_token() -> None:
    with TestClient(app) as client:
        response = _post(client, INITIALIZE, token=None)
        assert response.status_code == 401


def test_mcp_rejects_requests_with_wrong_token() -> None:
    with TestClient(app) as client:
        response = _post(client, INITIALIZE, token="wrong-token")
        assert response.status_code == 401


def test_mcp_initialize_and_list_tools() -> None:
    with TestClient(app) as client:
        session_id = _initialize(client)

        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            headers={
                "Accept": ACCEPT,
                "Authorization": f"Bearer {TOKEN}",
                "mcp-session-id": session_id,
            },
        )
        assert response.status_code == 200
        assert "search_knowledge_base" in response.text
        assert "list_documents" in response.text


# ----------------------------------------------------------------------
# Tool behaviour (unit tests — pgvector SQL is not available on sqlite)
# ----------------------------------------------------------------------


class _FakeRow:
    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def _fake_session(rows: list[object]) -> MagicMock:
    """A session whose ``execute`` is awaitable and returns ``rows``."""
    session = MagicMock()
    execute = AsyncMock()
    result = MagicMock()
    result.fetchall.return_value = rows
    execute.return_value = result
    session.execute = execute
    return session


@patch("app.mcp.tools.get_embedding_provider")
@patch("app.mcp.tools.async_session_factory")
async def test_search_knowledge_base_returns_results(
    mock_factory: MagicMock, mock_get_provider: MagicMock
) -> None:
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_get_provider.return_value = embedder

    rows = [
        _FakeRow(
            chunk_id="c1", content="first chunk", document_id="doc-1", filename="a.txt", score=0.1
        ),
        _FakeRow(
            chunk_id="c2", content="second chunk", document_id="doc-2", filename="b.pdf", score=0.4
        ),
    ]
    session = _fake_session(rows)
    mock_factory.return_value.__aenter__.return_value = session

    results = await search_knowledge_base("some question", top_k=5)

    assert [r.chunk_id for r in results] == ["c1", "c2"]
    assert results[0].filename == "a.txt"
    assert results[1].score == 0.4

    _, params = session.execute.call_args[0]
    assert params["limit"] == 5
    assert params["query_vector"] == "[0.1, 0.2, 0.3]"


@patch("app.mcp.tools.get_embedding_provider")
@patch("app.mcp.tools.async_session_factory")
async def test_search_knowledge_base_clamps_top_k(
    mock_factory: MagicMock, mock_get_provider: MagicMock
) -> None:
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[[0.0]])
    mock_get_provider.return_value = embedder
    session = _fake_session([])
    mock_factory.return_value.__aenter__.return_value = session

    await search_knowledge_base("q", top_k=1000)

    _, params = session.execute.call_args[0]
    assert params["limit"] == 20


@patch("app.mcp.tools.async_session_factory")
async def test_list_documents(mock_factory: MagicMock) -> None:
    rows = [
        _FakeRow(
            id="doc-1", filename="a.txt", content_type="text/plain", created_at=datetime(2026, 1, 1)
        ),
    ]
    session = _fake_session(rows)
    mock_factory.return_value.__aenter__.return_value = session

    documents = await list_documents()

    assert len(documents) == 1
    assert documents[0].document_id == "doc-1"
    assert documents[0].created_at == datetime(2026, 1, 1)
