import json
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.session import get_db_session
from app.models.user import User
from app.schemas.chat import ChatCompletionRequest
from app.services.chat_service import ChatService

router = APIRouter()


@router.post("/chat/completions")
async def create_chat_completion(
    payload: ChatCompletionRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    service = ChatService(session)

    if payload.stream:
        return StreamingResponse(
            _stream_events(service, payload, current_user),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return await service.create_completion(payload, current_user)


async def _stream_events(
    service: ChatService,
    payload: ChatCompletionRequest,
    user: User,
):
    """Yield SSE-formatted bytes for each chunk."""
    async for chunk in service.create_completion_stream(payload, user):
        data = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk.content},
                    "finish_reason": chunk.finish_reason,
                }
            ],
        }
        if chunk.conversation_id:
            data["conversation_id"] = chunk.conversation_id
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
