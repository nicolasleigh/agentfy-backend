"""Outbound MCP client manager (Direction A).

Connects to the MCP servers configured in ``settings.mcp_servers``, lists
their tools, converts them to OpenAI-compatible LLM tool definitions, and
dispatches tool calls back to the right server. A module-level singleton is
connected/closed from the FastAPI lifespan.
"""

import json
import logging

from fastmcp import Client
from mcp.types import TextContent, Tool

from app.core.config import settings

logger = logging.getLogger(__name__)


def _tool_to_llm_schema(tool: Tool, full_name: str) -> dict:
    """Convert an MCP tool into an OpenAI-compatible tool definition.

    MCP's ``inputSchema`` is already a JSON-Schema ``parameters`` object, so
    it maps directly onto the LLM function schema.
    """
    return {
        "type": "function",
        "function": {
            "name": full_name,
            "description": tool.description,
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


def _result_to_text(result) -> str:
    """Extract a plain-text result from a ``CallToolResult``."""
    parts = [b.text for b in result.content if isinstance(b, TextContent)]
    if parts:
        return "\n".join(parts)
    if getattr(result, "structured_content", None):
        return json.dumps(result.structured_content, ensure_ascii=False)
    if getattr(result, "data", None) is not None:
        return json.dumps(result.data, ensure_ascii=False, default=str)
    return str(result)


class MCPClientManager:
    """Holds persistent connections to all configured MCP servers."""

    def __init__(self) -> None:
        self._clients: dict[str, Client] = {}
        # full tool name (``<server>:<tool>``) -> (server name, tool name)
        self._registry: dict[str, tuple[str, str]] = {}
        self._tools: list[dict] = []

    @property
    def is_connected(self) -> bool:
        return bool(self._clients)

    async def connect(self) -> None:
        """Connect to every configured server and register its tools.

        A server that fails to connect is skipped with a warning (tool calling
        simply won't include its tools) rather than crashing the app.
        """
        for cfg in settings.mcp_servers:
            if cfg.name in self._clients:
                continue
            try:
                client = Client(cfg.url, auth=cfg.auth_token)
                await client._connect()  # noqa: SLF001 — FastMCP has no public connect
            except Exception as e:
                logger.warning("Failed to connect to MCP server '%s': %s", cfg.name, e)
                continue

            tools = await client.list_tools()
            for tool in tools:
                full_name = f"{cfg.name}:{tool.name}"
                self._registry[full_name] = (cfg.name, tool.name)
                self._tools.append(_tool_to_llm_schema(tool, full_name))
            self._clients[cfg.name] = client
            logger.info("Connected to MCP server '%s' (%d tools)", cfg.name, len(tools))

    async def close(self) -> None:
        for name, client in self._clients.items():
            try:
                await client.close()
            except Exception:
                logger.warning("Error closing MCP client '%s'", name, exc_info=True)
        self._clients.clear()
        self._registry.clear()
        self._tools.clear()

    def get_external_tools(self) -> list[dict]:
        """LLM tool definitions for all connected servers' tools."""
        return list(self._tools)

    async def call_tool(self, full_name: str, arguments: dict) -> str:
        """Execute a tool on its server and return the result as text."""
        entry = self._registry.get(full_name)
        if entry is None:
            return f"Error: unknown tool '{full_name}'"
        server_name, tool_name = entry
        client = self._clients.get(server_name)
        if client is None:
            return f"Error: MCP server '{server_name}' is not connected"

        try:
            result = await client.call_tool(tool_name, arguments, raise_on_error=False)
        except Exception as e:  # network / protocol errors
            return f"Error calling tool '{full_name}': {e}"
        if result.is_error:
            return f"Error calling tool '{full_name}': {_result_to_text(result)}"
        return _result_to_text(result)


mcp_client_manager = MCPClientManager()
