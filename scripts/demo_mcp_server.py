"""A tiny MCP server used to exercise the backend's MCP client (Direction A).

Runs a Streamable-HTTP MCP server on http://localhost:9100/mcp with a couple
of toy tools. Point ``MCP_SERVERS`` at it in ``.env``:

    MCP_SERVERS=[{"name":"demo","url":"http://localhost:9100/mcp"}]

Run with:  .venv/bin/python scripts/demo_mcp_server.py
"""

from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastmcp import FastMCP

mcp = FastMCP("demo")


@mcp.tool()
async def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()


@mcp.tool()
async def echo(text: str) -> str:
    """Echo back the given text."""
    return f"echo: {text}"


if __name__ == "__main__":
    mcp_app = mcp.http_app(transport="streamable-http", path="/")
    app = FastAPI(lifespan=mcp_app.lifespan)
    app.mount("/mcp", mcp_app)
    uvicorn.run(app, host="0.0.0.0", port=9100)
