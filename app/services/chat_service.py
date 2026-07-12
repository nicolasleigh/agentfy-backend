from datetime import UTC, datetime
from uuid import uuid4

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_completion import ChatCompletion
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User
from app.providers import get_llm_provider, LLMResult
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
        self.llm = get_llm_provider()

    async def create_completion(
        self,
        payload: ChatCompletionRequest,
        user: User,
    ) -> ChatCompletionResponse:
        now = datetime.now(UTC)
        completion_id = f"chatcmpl-{uuid4().hex}"

        # Call the LLM provider
        try:
            result: LLMResult = await self.llm.chat_completion(
                model=payload.model,
                messages=[m.model_dump() for m in payload.messages],
                temperature=payload.temperature,
                stream=payload.stream,
            )
        except ConnectionError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            ) from e
        except TimeoutError as e:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=str(e),
            ) from e
        except RuntimeError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(e),
            ) from e

        assistant_message = ChatMessage(
            role=result.role,
            content=result.content,
        )
        response = ChatCompletionResponse(
            id=completion_id,
            created=result.created or int(now.timestamp()),
            model=result.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=assistant_message,
                    finish_reason=result.finish_reason,
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
            ),
        )

        # Resolve conversation: use existing or auto-create from the first user message
        conversation = await self._resolve_conversation(payload, user)

        # Persist the raw completion record
        record = ChatCompletion(
            id=completion_id,
            user_id=user.id,
            conversation_id=conversation.id if conversation else None,
            model=response.model,
            request=payload.model_dump(mode="json"),
            response=response.model_dump(mode="json"),
            created_at=now,
        )
        self.session.add(record)

        # Persist individual messages
        if conversation:
            self._persist_messages(payload, assistant_message, conversation.id, now)

        await self.session.commit()

        return response

    def _persist_messages(
        self,
        payload: ChatCompletionRequest,
        assistant_message: ChatMessage,
        conversation_id: str,
        now: datetime,
    ) -> None:
        # Write all user messages from the request
        for msg in payload.messages:
            if msg.role == "user":
                self.session.add(
                    Message(
                        id=f"msg-{uuid4().hex}",
                        conversation_id=conversation_id,
                        role=msg.role,
                        content=msg.content,
                        created_at=now,
                    )
                )

        # Write the assistant response
        self.session.add(
            Message(
                id=f"msg-{uuid4().hex}",
                conversation_id=conversation_id,
                role=assistant_message.role,
                content=assistant_message.content,
                created_at=now,
            )
        )

    async def _resolve_conversation(
        self,
        payload: ChatCompletionRequest,
        user: User,
    ) -> Conversation | None:
        if payload.conversation_id:
            # Verify the conversation exists and belongs to this user
            result = await self.session.execute(
                select(Conversation).where(
                    Conversation.id == payload.conversation_id,
                    Conversation.user_id == user.id,
                )
            )
            conversation = result.scalar_one_or_none()
            if conversation is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found",
                )
            # Touch updated_at
            conversation.updated_at = datetime.now(UTC)
            self.session.add(conversation)
            return conversation

        # Auto-create a conversation from the first user message
        first_user_msg = next(
            (msg.content for msg in payload.messages if msg.role == "user"),
            "New Chat",
        )
        title = first_user_msg[:255] if first_user_msg else "New Chat"
        conversation = Conversation(
            id=f"conv-{uuid4().hex}",
            user_id=user.id,
            title=title,
        )
        self.session.add(conversation)
        return conversation
