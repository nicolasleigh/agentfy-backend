from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.mcp.client import mcp_client_manager
from app.mcp.server import create_mcp_server


@asynccontextmanager
async def _app_lifespan(app: FastAPI):
    """Compose the MCP server's session manager with outbound MCP clients.

    ``mcp_app.lifespan`` runs FastMCP's StreamableHTTPSessionManager; we also
    connect the outbound MCP client manager (Direction A) and close it on exit.
    """
    mcp_app = app.state.mcp_app
    async with mcp_app.lifespan(app):
        await mcp_client_manager.connect()
        try:
            yield
        finally:
            await mcp_client_manager.close()


def create_app() -> FastAPI:
    # MCP server (Direction B): knowledge-base search over Streamable HTTP.
    # Mounted at /mcp; its lifespan runs the MCP session manager.
    mcp_app = create_mcp_server().http_app(transport="streamable-http", path="/")

    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        version="0.1.0",
        lifespan=_app_lifespan,
    )
    app.state.mcp_app = mcp_app

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.backend_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.api_v1_prefix)
    app.mount("/mcp", mcp_app)

    @app.get("/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
