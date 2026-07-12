from datetime import UTC, datetime
from uuid import uuid4

import httpx
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.chat_completion import ChatCompletion
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User
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
        self.ollama_base_url = settings.ollama_base_url.rstrip("/")

    async def create_completion(
        self,
        payload: ChatCompletionRequest,
        user: User,
    ) -> ChatCompletionResponse:
        now = datetime.now(UTC)
        completion_id = f"chatcmpl-{uuid4().hex}"

        # Call the Ollama API
        ollama_response = await self._call_ollama(payload)

        # Map Ollama response into our schema
        ollama_data = ollama_response
        choice = ollama_data["choices"][0]
        assistant_message = ChatMessage(
            role=choice["message"]["role"],
            content=choice["message"]["content"],
        )
        usage_data = ollama_data.get("usage", {})
        response = ChatCompletionResponse(
            id=completion_id,
            created=ollama_data.get("created", int(now.timestamp())),
            model=ollama_data.get("model", payload.model),
            choices=[
                ChatCompletionChoice(
                    index=choice.get("index", 0),
                    message=assistant_message,
                    finish_reason=choice.get("finish_reason", "stop"),
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
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

    async def _call_ollama(
        self,
        payload: ChatCompletionRequest,
    ) -> dict:
        """Call Ollama's OpenAI-compatible /v1/chat/completions endpoint."""
        url = f"{self.ollama_base_url}/v1/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    url,
                    json={
                        "model": payload.model,
                        "messages": [m.model_dump() for m in payload.messages],
                        "temperature": payload.temperature,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Cannot connect to Ollama at {self.ollama_base_url}. "
                       f"Is Ollama running? (Error: {e})",
            ) from e
        except httpx.TimeoutException as e:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Ollama request timed out. The model may still be loading.",
            ) from e
        except httpx.HTTPStatusError as e:
            detail = f"Ollama returned an error: {e.response.status_code}"
            try:
                body = e.response.json()
                if "error" in body:
                    detail += f" - {body['error']}"
            except Exception:
                pass
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=detail,
            ) from e

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
