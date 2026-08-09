"""Tool implementations for the MCP server.

These search the whole knowledge base (all users) — the MCP surface is an
app-level integration, and the document corpus is shared. The pgvector
queries mirror ``app.services.embedding_service.EmbeddingService.retrieve``
but drop the ``WHERE d.user_id`` filter.
"""

from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import text

from app.db.session import async_session_factory
from app.providers import get_embedding_provider


class KnowledgeBaseSearchResult(BaseModel):
    """A single retrieved chunk with its source document."""

    chunk_id: str
    content: str
    document_id: str
    filename: str
    score: float


class DocumentInfo(BaseModel):
    """Metadata for a document in the knowledge base."""

    document_id: str
    filename: str
    content_type: str
    created_at: datetime


async def search_knowledge_base(
    query: str,
    top_k: int = 5,
) -> list[KnowledgeBaseSearchResult]:
    """Vector-similarity search across the entire knowledge base.

    Returns the ``top_k`` document chunks closest to ``query`` (by cosine
    distance over the pgvector embeddings), each with its content, source
    document, and score.
    """
    top_k = max(1, min(top_k, 20))

    query_vector = (await get_embedding_provider().embed([query]))[0]

    sql = text("""
        SELECT
            dc.id          AS chunk_id,
            dc.content,
            dc.document_id,
            d.filename,
            (dc.embedding <=> CAST(:query_vector AS vector)) AS score
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        ORDER BY score ASC
        LIMIT :limit
    """)

    async with async_session_factory() as session:
        result = await session.execute(
            sql,
            {
                "query_vector": str(query_vector),
                "limit": top_k,
            },
        )
        rows = result.fetchall()

    return [
        KnowledgeBaseSearchResult(
            chunk_id=row.chunk_id,
            content=row.content,
            document_id=row.document_id,
            filename=row.filename,
            score=float(row.score),
        )
        for row in rows
    ]


async def list_documents() -> list[DocumentInfo]:
    """List all documents currently in the knowledge base."""
    sql = text("""
        SELECT id, filename, content_type, created_at
        FROM documents
        ORDER BY created_at DESC
    """)

    async with async_session_factory() as session:
        result = await session.execute(sql)
        rows = result.fetchall()

    return [
        DocumentInfo(
            document_id=row.id,
            filename=row.filename,
            content_type=row.content_type,
            created_at=row.created_at,
        )
        for row in rows
    ]
