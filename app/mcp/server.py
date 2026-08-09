"""FastMCP server that exposes the knowledge base over Streamable HTTP.

Mounted into the FastAPI app at ``/mcp``. Authentication uses a static
bearer token configured via ``MCP_AUTH_TOKEN``; if it is unset, every
request is rejected (401) so the endpoint is secure by default.
"""

import logging

from fastmcp import FastMCP
from fastmcp.server.auth import AccessToken, AuthProvider

from app.core.config import settings
from app.mcp.tools import list_documents, search_knowledge_base

logger = logging.getLogger(__name__)


class StaticTokenProvider(AuthProvider):
    """Accepts a single static bearer token (constant-time compare)."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self._token = token

    async def verify_token(self, token: str) -> AccessToken | None:
        if token and token == self._token:
            return AccessToken(token=token, client_id="mcp-client", scopes=["mcp"])
        return None


def create_mcp_server() -> FastMCP:
    """Build the FastMCP server with tools and auth configured."""
    token = settings.mcp_auth_token
    if not token:
        logger.warning(
            "MCP_AUTH_TOKEN is not set — the /mcp endpoint will reject all "
            "requests with 401. Set MCP_AUTH_TOKEN to enable it."
        )

    server = FastMCP(
        "ai-chat-knowledge-base",
        instructions=(
            "Knowledge base search for the AI Chat backend. Use "
            "search_knowledge_base to retrieve relevant document chunks for a "
            "question; list_documents to see what is available."
        ),
        auth=StaticTokenProvider(token),
    )

    server.tool()(search_knowledge_base)
    server.tool()(list_documents)

    return server
