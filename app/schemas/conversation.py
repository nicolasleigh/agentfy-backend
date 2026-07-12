from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.chat import ChatRole


class CreateConversationRequest(BaseModel):
    title: str = Field(default="New Chat", max_length=255, min_length=1)


class UpdateConversationRequest(BaseModel):
    title: str = Field(max_length=255, min_length=1)


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    conversations: list[ConversationResponse]
    total: int


class MessageResponse(BaseModel):
    id: str
    role: ChatRole
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationMessagesResponse(BaseModel):
    messages: list[MessageResponse]
    total: int
