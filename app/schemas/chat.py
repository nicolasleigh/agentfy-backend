from typing import Annotated, Literal

from pydantic import BaseModel, Field

ChatRole = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    role: ChatRole
    content: str


class ChatCompletionRequest(BaseModel):
    model: Annotated[str, Field(default="demo-chat", min_length=1)]
    messages: Annotated[list[ChatMessage], Field(min_length=1)]
    temperature: Annotated[float | None, Field(default=1.0, ge=0, le=2)]
    stream: bool = False
    conversation_id: str | None = None
    rag_enabled: bool = True
    # When True, the model may call tools (internal RAG search + external MCP
    # tools). Off by default for backward compatibility.
    tools_enabled: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
