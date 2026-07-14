from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.user import User
from app.schemas.document import DocumentListResponse, DocumentResponse
from app.services.embedding_service import EmbeddingService

# Default chunk parameters
_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 64


class DocumentService:
    """Handle document upload, text extraction, chunking, and embedding."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.embedding_service = EmbeddingService(session)

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    async def upload(
        self,
        filename: str,
        content_type: str,
        raw: bytes,
        user: User,
    ) -> DocumentResponse:
        """Parse, chunk, embed, and persist a document."""

        # 1. Extract text
        text = _extract_text(raw, content_type)

        # 2. Chunk
        chunks = _chunk_text(text, chunk_size=_CHUNK_SIZE, overlap=_CHUNK_OVERLAP)

        # 3. Create Document record
        document = Document(
            id=f"doc-{uuid4().hex}",
            user_id=user.id,
            filename=filename,
            content_type=content_type,
        )
        self.session.add(document)

        # 4. Embed and store chunks (will flush within the method)
        if chunks:
            await self.embedding_service.embed_and_store_chunks(document, chunks)

        await self.session.commit()
        await self.session.refresh(document)

        return DocumentResponse.model_validate(document)

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    async def list_documents(self, user: User) -> DocumentListResponse:
        result = await self.session.execute(
            select(Document)
            .where(Document.user_id == user.id)
            .order_by(Document.created_at.desc())
        )
        documents = result.scalars().all()
        return DocumentListResponse(
            documents=[DocumentResponse.model_validate(d) for d in documents],
            total=len(documents),
        )

    # ------------------------------------------------------------------
    # Get single
    # ------------------------------------------------------------------

    async def get_document(self, document_id: str, user: User) -> DocumentResponse:
        document = await self._get_owned(document_id, user)
        return DocumentResponse.model_validate(document)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_document(self, document_id: str, user: User) -> None:
        document = await self._get_owned(document_id, user)
        await self.session.delete(document)
        await self.session.commit()

    async def _get_owned(self, document_id: str, user: User) -> Document:
        result = await self.session.execute(
            select(Document).where(
                Document.id == document_id,
                Document.user_id == user.id,
            )
        )
        document = result.scalar_one_or_none()
        if document is None:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )
        return document


# ------------------------------------------------------------------
# Text extraction
# ------------------------------------------------------------------

_SUPPORTED_TEXT_TYPES = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "application/xml",
}


def _extract_text(raw: bytes, content_type: str) -> str:
    """Extract plain text from raw bytes based on content type."""
    if content_type == "application/pdf":
        return _extract_pdf_text(raw)
    if content_type in _SUPPORTED_TEXT_TYPES:
        return raw.decode("utf-8", errors="replace")
    # Fallback: try as plain text
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(
            f"Unsupported content type '{content_type}' and "
            f"binary content cannot be decoded as UTF-8."
        )


def _extract_pdf_text(raw: bytes) -> str:
    """Extract text from a PDF using PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=raw, filetype="pdf")
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


# ------------------------------------------------------------------
# Text chunking
# ------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks by character count.

    Preserves as much paragraph structure as possible by preferring
    ``\\n\\n`` or ``\\n`` boundaries near the target size.
    """
    if not text.strip():
        return []

    # Normalise line endings
    text = text.replace("\r\n", "\n")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to break at paragraph boundary (double newline) before ``end``
        para_break = text.rfind("\n\n", start, end)
        if para_break > start + chunk_size // 2:
            end = para_break + 2
        else:
            # Try single newline
            line_break = text.rfind("\n", start, end)
            if line_break > start + chunk_size // 2:
                end = line_break + 1

        chunks.append(text[start:end])
        start = end - overlap

    return chunks
