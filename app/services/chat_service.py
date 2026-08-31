import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.mcp.agent import run_tool_loop
from app.mcp.client import mcp_client_manager
from app.models.chat_completion import ChatCompletion
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User
from app.providers import LLMResult, LLMStreamChunk, get_llm_provider
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
            if payload.tools_enabled:
                tools, tool_executor = await self._build_tools(user)
                if tools:
                    result = await run_tool_loop(
                        self.llm,
                        messages,
                        payload.model,
                        payload.temperature,
                        tools,
                        tool_executor,
                    )
                else:
                    result = await self.llm.chat_completion(
                        model=payload.model,
                        messages=messages,
                        temperature=payload.temperature,
                    )
            else:
                result = await self.llm.chat_completion(
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

        # Agentic tool path: run the tool loop in a task, emitting a
        # ``tool_call`` SSE event as each tool starts, then the final answer
        # in chunks once the loop finishes.
        agentic_result: LLMResult | None = None
        if payload.tools_enabled:
            tools, tool_executor = await self._build_tools(user)
            if tools:
                events: asyncio.Queue[dict] = asyncio.Queue()

                async def _on_tool(name: str, arguments: dict) -> None:
                    events.put_nowait({"type": "tool_call", "name": name})

                async def _run_loop() -> None:
                    try:
                        result = await run_tool_loop(
                            self.llm,
                            messages,
                            payload.model,
                            payload.temperature,
                            tools,
                            tool_executor,
                            on_tool=_on_tool,
                        )
                        await events.put({"type": "result", "result": result})
                    except Exception as e:  # noqa: BLE001 — surfaced to the stream
                        await events.put({"type": "error", "error": e})

                task = asyncio.create_task(_run_loop())
                try:
                    while True:
                        event = await events.get()
                        if event["type"] == "tool_call":
                            yield LLMStreamChunk(tool_call=event["name"])
                        elif event["type"] == "error":
                            e = event["error"]
                            if isinstance(e, ConnectionError):
                                raise HTTPException(
                                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                    detail=str(e),
                                ) from e
                            if isinstance(e, TimeoutError):
                                raise HTTPException(
                                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                                    detail=str(e),
                                ) from e
                            logging.getLogger(__name__).exception(
                                "Agentic loop failed", exc_info=e
                            )
                            raise HTTPException(
                                status_code=status.HTTP_502_BAD_GATEWAY,
                                detail=str(e),
                            ) from e
                        else:
                            agentic_result = event["result"]
                            break
                finally:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

        # Accumulate the full reply as chunks arrive
        full_content: list[str] = []
        final_reason: str = "stop"

        if agentic_result is not None:
            content = agentic_result.content
            final_reason = agentic_result.finish_reason or "stop"
            for i in range(0, len(content), 200):
                yield LLMStreamChunk(content=content[i:i + 200])
            yield LLMStreamChunk(finish_reason=final_reason)
        else:
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
            content = "".join(full_content)

        # Build the full response for persistence
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
    # Tool calling (Direction A)
    # ------------------------------------------------------------------

    async def _build_tools(
        self,
        user: User,
    ) -> tuple[list[dict], Callable[[str, dict], Awaitable[str]]]:
        """Build LLM tool definitions and the executor that dispatches calls.

        Always exposes a user-scoped knowledge-base search tool; additionally
        exposes every tool from the configured external MCP servers.
        """
        tools: list[dict] = []

        async def search_knowledge_base(query: str, top_k: int = 5) -> str:
            """User-scoped RAG search (mirrors the automatic RAG injection)."""
            try:
                emb_service = EmbeddingService(self.session)
                chunks = await emb_service.retrieve(query, user, top_k=max(1, min(top_k, 20)))
            except Exception:
                await self.session.rollback()
                return "Knowledge base search failed."
            if not chunks:
                return "No relevant documents found in the knowledge base."
            return json.dumps(
                [
                    {
                        "filename": c.get("filename", ""),
                        "content": c.get("content", ""),
                        "score": c.get("score", 0),
                    }
                    for c in chunks
                ],
                ensure_ascii=False,
            )

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": (
                        "Search the user's knowledge base for document chunks "
                        "relevant to a question."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"},
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return (1-20)",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        )
        tools.extend(mcp_client_manager.get_external_tools())

        async def executor(name: str, arguments: dict) -> str:
            if name == "search_knowledge_base":
                return await search_knowledge_base(**arguments)
            return await mcp_client_manager.call_tool(name, arguments)

        return tools, executor

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
