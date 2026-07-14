from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.session import get_db_session
from app.models.user import User
from app.schemas.document import DocumentListResponse, DocumentResponse
from app.services.document_service import DocumentService

router = APIRouter()


@router.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> DocumentResponse:
    """Upload a document (PDF, TXT, MD, CSV, JSON, XML) for RAG indexing."""
    raw = await file.read()
    service = DocumentService(session)
    return await service.upload(
        filename=file.filename or "unnamed",
        content_type=file.content_type or "text/plain",
        raw=raw,
        user=current_user,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> DocumentListResponse:
    """List all documents uploaded by the current user."""
    service = DocumentService(session)
    return await service.list_documents(current_user)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> DocumentResponse:
    """Get a single document's metadata."""
    service = DocumentService(session)
    return await service.get_document(document_id, current_user)


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(
    document_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> Response:
    """Delete a document and all its chunks."""
    service = DocumentService(session)
    await service.delete_document(document_id, current_user)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
