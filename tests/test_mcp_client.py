"""Tests for the MCP client (Direction A: model tool calling)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from mcp.types import TextContent, Tool

from app.main import app
from app.mcp.agent import run_tool_loop
from app.mcp.client import MCPClientManager, _tool_to_llm_schema
from app.providers.base import LLMResult


def _result(content: str = "", tool_calls: list | None = None) -> LLMResult:
    return LLMResult(content=content, model="m", created=0, tool_calls=tool_calls or [])


def _tool_call(name: str, arguments: dict) -> dict:
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class _ScriptedLLM:
    """A fake provider returning scripted responses; records its calls."""

    def __init__(self, responses: list[LLMResult]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple] = []

    async def chat_completion(
        self, model: str, messages: list[dict], temperature=None, tools=None
    ) -> LLMResult:
        self.calls.append((model, list(messages), temperature, tools))
        return self.responses.pop(0)


# ----------------------------------------------------------------------
# Schema conversion
# ----------------------------------------------------------------------


def test_tool_to_llm_schema() -> None:
    tool = Tool(
        name="add",
        description="Add two numbers.",
        inputSchema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )
    out = _tool_to_llm_schema(tool, "demo:add")

    assert out["type"] == "function"
    fn = out["function"]
    assert fn["name"] == "demo:add"
    assert fn["description"] == "Add two numbers."
    assert fn["parameters"] == tool.inputSchema


# ----------------------------------------------------------------------
# Client manager dispatch
# ----------------------------------------------------------------------


def test_get_external_tools_returns_a_copy() -> None:
    manager = MCPClientManager()
    manager._tools = [{"type": "function"}]
    out = manager.get_external_tools()
    out.append({"x": 1})
    assert manager._tools == [{"type": "function"}]


@pytest.mark.asyncio
async def test_call_tool_dispatches_to_server() -> None:
    manager = MCPClientManager()
    fake = MagicMock()
    fake.call_tool = AsyncMock(
        return_value=MagicMock(is_error=False, content=[TextContent(type="text", text="42")])
    )
    manager._clients["demo"] = fake
    manager._registry["demo:add"] = ("demo", "add")

    out = await manager.call_tool("demo:add", {"a": 1, "b": 2})

    assert out == "42"
    fake.call_tool.assert_awaited_once_with("add", {"a": 1, "b": 2}, raise_on_error=False)


@pytest.mark.asyncio
async def test_call_tool_unknown_tool() -> None:
    manager = MCPClientManager()
    out = await manager.call_tool("nope", {})
    assert "unknown tool" in out


@pytest.mark.asyncio
async def test_call_tool_surfaces_tool_error() -> None:
    manager = MCPClientManager()
    fake = MagicMock()
    fake.call_tool = AsyncMock(
        return_value=MagicMock(is_error=True, content=[TextContent(type="text", text="boom")])
    )
    manager._clients["demo"] = fake
    manager._registry["demo:bad"] = ("demo", "bad")

    out = await manager.call_tool("demo:bad", {})
    assert "boom" in out


# ----------------------------------------------------------------------
# Agentic loop
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_tool_loop_executes_tool_then_answers() -> None:
    tool_call = _tool_call("demo:add", {"a": 1, "b": 2})
    llm = _ScriptedLLM([_result(tool_calls=[tool_call]), _result(content="the answer is 3")])

    executed: list[tuple] = []

    async def executor(name: str, arguments: dict) -> str:
        executed.append((name, arguments))
        return "3"

    messages = [{"role": "user", "content": "add 1 and 2"}]
    result = await run_tool_loop(llm, messages, "m", None, [{"type": "function"}], executor)

    assert result.content == "the answer is 3"
    assert executed == [("demo:add", {"a": 1, "b": 2})]
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["tool_calls"] == [tool_call]
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["content"] == "3"


@pytest.mark.asyncio
async def test_run_tool_loop_feeds_tool_error_back_to_model() -> None:
    llm = _ScriptedLLM([_result(tool_calls=[_tool_call("boom", {})]), _result(content="sorry")])

    async def executor(name: str, arguments: dict) -> str:
        raise RuntimeError("kaboom")

    messages = [{"role": "user", "content": "hi"}]
    result = await run_tool_loop(llm, messages, "m", None, [{}], executor)

    assert result.content == "sorry"
    assert "kaboom" in messages[-1]["content"]


@pytest.mark.asyncio
async def test_run_tool_loop_stops_at_max_iterations() -> None:
    llm = _ScriptedLLM([_result(tool_calls=[_tool_call("x", {})])] * 10)

    async def executor(name: str, arguments: dict) -> str:
        return "ok"

    await run_tool_loop(
        llm, [{"role": "user", "content": "hi"}], "m", None, [{}], executor, max_iterations=3
    )

    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_run_tool_loop_invokes_on_tool() -> None:
    llm = _ScriptedLLM(
        [_result(tool_calls=[_tool_call("demo:get_current_time", {})]), _result(content="done")]
    )

    observed: list[str] = []

    async def executor(name: str, arguments: dict) -> str:
        return "2026-01-01"

    async def on_tool(name: str, arguments: dict) -> None:
        observed.append(name)

    await run_tool_loop(
        llm, [{"role": "user", "content": "time"}], "m", None, [{}], executor, on_tool=on_tool
    )

    assert observed == ["demo:get_current_time"]


# ----------------------------------------------------------------------
# Chat endpoint integration (tools_enabled)
# ----------------------------------------------------------------------


def _register_and_login(client: TestClient) -> dict:
    email = f"{uuid4().hex}@example.com"
    client.post(
        "/v1/auth/register",
        json={"email": email, "password": "password123", "name": "tools"},
    )
    login = client.post("/v1/auth/login", json={"email": email, "password": "password123"})
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@patch("app.services.chat_service.get_llm_provider")
def test_chat_completion_runs_tool_loop_when_enabled(mock_get_provider: MagicMock) -> None:
    # First call: the model wants to search the KB. Second: final answer.
    scripted = _ScriptedLLM(
        [
            _result(tool_calls=[_tool_call("search_knowledge_base", {"query": "pgvector"})]),
            _result(content="Final answer referencing the search."),
        ]
    )
    mock_get_provider.return_value = scripted

    client = TestClient(app)
    headers = _register_and_login(client)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "demo-chat",
            "tools_enabled": True,
            "messages": [{"role": "user", "content": "查一下知识库里的 pgvector 是什么"}],
        },
        headers=headers,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Final answer referencing the search."
    # The loop made two provider calls (tool turn + final turn).
    assert len(scripted.calls) == 2


def test_chat_completion_without_tools_makes_single_call() -> None:
    # tools_enabled defaults to False → provider called exactly once, no loop.
    scripted = _ScriptedLLM([_result(content="plain answer")])

    client = TestClient(app)
    headers = _register_and_login(client)

    with patch("app.services.chat_service.get_llm_provider", return_value=scripted):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "demo-chat",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=headers,
        )

    assert resp.status_code == 200
    assert len(scripted.calls) == 1


@patch("app.services.chat_service.get_llm_provider")
def test_chat_stream_emits_tool_call_event(mock_get_provider: MagicMock) -> None:
    scripted = _ScriptedLLM(
        [
            _result(tool_calls=[_tool_call("search_knowledge_base", {"query": "pgvector"})]),
            _result(content="streamed final answer"),
        ]
    )
    mock_get_provider.return_value = scripted

    client = TestClient(app)
    headers = _register_and_login(client)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "demo-chat",
            "tools_enabled": True,
            "stream": True,
            "messages": [{"role": "user", "content": "查一下 pgvector"}],
        },
        headers=headers,
    )

    assert resp.status_code == 200
    assert '"tool_call": "search_knowledge_base"' in resp.text
    assert "streamed final answer" in resp.text
