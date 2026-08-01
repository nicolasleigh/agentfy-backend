from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_completion import ChatCompletion
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User
from app.providers import get_llm_provider, LLMResult, LLMStreamChunk
from app.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from app.services.embedding_service import EmbeddingService


class ChatService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.llm = get_llm_provider()

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def create_completion(
        self,
        payload: ChatCompletionRequest,
        user: User,
    ) -> ChatCompletionResponse:
        now = datetime.now(UTC)
        completion_id = f"chatcmpl-{uuid4().hex}"

        # RAG enrichment: retrieve context and inject into messages
        messages = await self._enrich_with_context(payload, user)

        try:
            result: LLMResult = await self.llm.chat_completion(
                model=payload.model,
                messages=messages,
                temperature=payload.temperature,
            )
        except ConnectionError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e),
            ) from e
        except TimeoutError as e:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(e),
            ) from e
        except RuntimeError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e),
            ) from e

        assistant_message = ChatMessage(role=result.role, content=result.content)
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

        conversation = await self._resolve_conversation(payload, user)
        await self._persist_completion(
            payload, response, assistant_message, conversation, completion_id, now, user,
        )

        return response

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def create_completion_stream(
        self,
        payload: ChatCompletionRequest,
        user: User,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Stream a completion, then persist the accumulated result.

        Yields ``LLMStreamChunk`` objects (the route layer formats them as SSE).
        After the last chunk, the completion record and messages are persisted.
        """
        now = datetime.now(UTC)
        completion_id = f"chatcmpl-{uuid4().hex}"

        conversation = await self._resolve_conversation(payload, user)

        # Lead with a meta chunk so the client learns the conversation id
        # (relevant when one was auto-created from a request without an id).
        yield LLMStreamChunk(conversation_id=conversation.id if conversation else None)

        # RAG enrichment: retrieve context and inject into messages
        messages = await self._enrich_with_context(payload, user, protected_objects=[conversation])

        # Accumulate the full reply as chunks arrive
        full_content: list[str] = []
        final_reason: str = "stop"
        try:
            async for chunk in self.llm.chat_completion_stream(
                model=payload.model,
                messages=messages,
                temperature=payload.temperature,
            ):
                if chunk.content:
                    full_content.append(chunk.content)
                if chunk.finish_reason:
                    final_reason = chunk.finish_reason
                yield chunk
        except ConnectionError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e),
            ) from e
        except TimeoutError as e:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(e),
            ) from e
        except RuntimeError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e),
            ) from e

        # Build the full response for persistence
        content = "".join(full_content)
        assistant_message = ChatMessage(role="assistant", content=content)
        response = ChatCompletionResponse(
            id=completion_id,
            created=int(now.timestamp()),
            model=payload.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=assistant_message,
                    finish_reason=final_reason,
                )
            ],
            usage=ChatCompletionUsage(),
        )

        # Persist after streaming completes
        try:
            await self._persist_completion(
                payload, response, assistant_message, conversation, completion_id, now, user,
            )
        except Exception:
            # Log but don't fail — SSE response headers have already been sent,
            # so raising here would cause ERR_INCOMPLETE_CHUNKED_ENCODING.
            import logging
            logging.getLogger(__name__).exception("Failed to persist chat completion")

    # ------------------------------------------------------------------
    # RAG enrichment
    # ------------------------------------------------------------------

    async def _enrich_with_context(
        self,
        payload: ChatCompletionRequest,
        user: User,
        *,
        protected_objects: list | None = None,
    ) -> list[dict]:
        """Retrieve relevant document chunks and inject them into the messages.

        If ``rag_enabled`` is ``False`` or no relevant context is found,
        the original messages are returned unchanged.
        """
        if not payload.rag_enabled:
            return [m.model_dump() for m in payload.messages]

        # Find the last user message to use as the retrieval query
        query = ""
        for msg in reversed(payload.messages):
            if msg.role == "user":
                query = msg.content
                break

        if not query:
            return [m.model_dump() for m in payload.messages]

        # Vector search
        try:
            emb_service = EmbeddingService(self.session)
            chunks = await emb_service.retrieve(query, user, top_k=5)
        except Exception:
            # If retrieval fails (e.g. no documents, DB error, missing pgvector),
            # rollback to clear PostgreSQL's aborted transaction state.
            # Expunge session-managed objects first to prevent rollback
            # from expiring their attributes (causes MissingGreenlet in SSE context).

            if protected_objects:
                for obj in protected_objects:
                    self.session.expunge(obj)
            self.session.expunge(user)
            await self.session.rollback()
            return [m.model_dump() for m in payload.messages]

        if not chunks:
            return [m.model_dump() for m in payload.messages]

        # Build context string
        context_parts: list[str] = []
        for chunk in chunks:
            source = chunk.get("filename", "unknown")
            content = chunk.get("content", "")
            context_parts.append(f"[{source}]: {content}")

        context_text = "\n\n".join(context_parts)

        system_prompt = (
            "You are a helpful assistant with access to the following reference material. "
            "Answer the user's question based on this context. "
            "If the context doesn't contain enough information, answer based on your own knowledge."
            f"\n\nContext:\n{context_text}"
        )

        return [
            {"role": "system", "content": system_prompt},
            *[m.model_dump() for m in payload.messages],
        ]

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    async def _persist_completion(
        self,
        payload: ChatCompletionRequest,
        response: ChatCompletionResponse,
        assistant_message: ChatMessage,
        conversation: Conversation | None,
        completion_id: str,
        now: datetime,
        user: User,
    ) -> None:
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

        if conversation:
            # Re-attach if it was expunged during RAG error recovery
            if conversation not in self.session:
                self.session.add(conversation)
            # 只存最后一条 user 消息，避免把历史中的 user 消息重复写入
            last_user_msg = next(
                (m for m in reversed(payload.messages) if m.role == "user"),
                None,
            )
            if last_user_msg:
                self.session.add(
                    Message(
                        id=f"msg-{uuid4().hex}",
                        conversation_id=conversation.id,
                        role=last_user_msg.role,
                        content=last_user_msg.content,
                        created_at=now,
                    )
                )
            self.session.add(
                Message(
                    id=f"msg-{uuid4().hex}",
                    conversation_id=conversation.id,
                    role=assistant_message.role,
                    content=assistant_message.content,
                    created_at=now,
                )
            )

        await self.session.commit()

    async def _resolve_conversation(
        self,
        payload: ChatCompletionRequest,
        user: User,
    ) -> Conversation | None:
        if payload.conversation_id:
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
            conversation.updated_at = datetime.now(UTC)
            self.session.add(conversation)
            return conversation

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
