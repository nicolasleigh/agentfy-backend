from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.user import User
from app.providers import get_embedding_provider


class EmbeddingService:
    """Handles embedding generation, storage, and similarity search."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.embedder = get_embedding_provider()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def embed_and_store_chunks(
        self,
        document: Document,
        chunks: list[str],
    ) -> list[DocumentChunk]:
        """Generate embeddings for text chunks and persist them.

        Args:
            document: The ``Document`` these chunks belong to.
            chunks: List of chunk text strings, in order.

        Returns:
            The list of persisted ``DocumentChunk`` records.
        """
        # Generate embeddings in batch
        embeddings = await self.embedder.embed(chunks)

        # Persist chunks
        records: list[DocumentChunk] = []
        for idx, (content, vector) in enumerate(zip(chunks, embeddings, strict=True)):
            record = DocumentChunk(
                id=f"chunk-{uuid4().hex}",
                document_id=document.id,
                chunk_index=idx,
                content=content,
                embedding=vector,
            )
            self.session.add(record)
            records.append(record)

        return records

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        user: User,
        top_k: int = 5,
    ) -> list[dict]:
        """Vector similarity search — find chunks closest to ``query``.

        Args:
            query: The user's question / search text.
            user: Only search chunks belonging to this user's documents.
            top_k: Number of results to return.

        Returns:
            List of ``{"chunk_id", "content", "document_id", "filename", "score"}``
            ordered by ascending cosine distance (closest first).
        """
        query_vector = (await self.embedder.embed([query]))[0]

        sql = text("""
            SELECT
                dc.id          AS chunk_id,
                dc.content,
                dc.document_id,
                d.filename,
                (dc.embedding <=> CAST(:query_vector AS vector)) AS score
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE d.user_id = :user_id
            ORDER BY score ASC
            LIMIT :limit
        """)

        result = await self.session.execute(
            sql,
            {
                "query_vector": str(query_vector),
                "user_id": user.id,
                "limit": top_k,
            },
        )
        rows = result.fetchall()

        return [
            {
                "chunk_id": row.chunk_id,
                "content": row.content,
                "document_id": row.document_id,
                "filename": row.filename,
                "score": float(row.score),
            }
            for row in rows
        ]
