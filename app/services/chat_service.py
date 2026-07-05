from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_completion import ChatCompletion
from app.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)


class ChatService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_completion(self, payload: ChatCompletionRequest) -> ChatCompletionResponse:
        now = datetime.now(UTC)
        completion_id = f"chatcmpl-{uuid4().hex}"
        assistant_message = ChatMessage(
            role="assistant",
            content=self._build_demo_reply(payload),
        )
        response = ChatCompletionResponse(
            id=completion_id,
            created=int(now.timestamp()),
            model=payload.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=assistant_message,
                )
            ],
            usage=ChatCompletionUsage(),
        )

        record = ChatCompletion(
            id=completion_id,
            model=payload.model,
            request=payload.model_dump(mode="json"),
            response=response.model_dump(mode="json"),
            created_at=now,
        )
        self.session.add(record)
        await self.session.commit()

        return response

    def _build_demo_reply(self, payload: ChatCompletionRequest) -> str:
        last_user_message = next(
            (message.content for message in reversed(payload.messages) if message.role == "user"),
            "",
        )
        if not last_user_message:
            return "Hello! Send a user message and I will echo it from the demo backend."
        return f"Demo backend received: {last_user_message}"
