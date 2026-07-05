from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.schemas.chat import ChatCompletionRequest, ChatCompletionResponse
from app.services.chat_service import ChatService

router = APIRouter()


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    payload: ChatCompletionRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ChatCompletionResponse:
    service = ChatService(session)
    return await service.create_completion(payload)
