from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User
from app.schemas.conversation import (
    ConversationListResponse,
    ConversationMessagesResponse,
    ConversationResponse,
    CreateConversationRequest,
    MessageResponse,
    UpdateConversationRequest,
)


class ConversationService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_conversation(
        self,
        payload: CreateConversationRequest,
        user: User,
    ) -> ConversationResponse:
        conversation = Conversation(
            id=f"conv-{uuid4().hex}",
            user_id=user.id,
            title=payload.title,
        )
        self.session.add(conversation)
        await self.session.commit()
        await self.session.refresh(conversation)
        return ConversationResponse.model_validate(conversation)

    async def list_conversations(
        self,
        user: User,
    ) -> ConversationListResponse:
        result = await self.session.execute(
            select(Conversation)
            .where(Conversation.user_id == user.id)
            .order_by(Conversation.updated_at.desc())
        )
        conversations = result.scalars().all()
        return ConversationListResponse(
            conversations=[
                ConversationResponse.model_validate(c) for c in conversations
            ],
            total=len(conversations),
        )

    async def get_conversation(
        self,
        conversation_id: str,
        user: User,
    ) -> ConversationResponse:
        conversation = await self._get_owned(conversation_id, user)
        return ConversationResponse.model_validate(conversation)

    async def update_conversation(
        self,
        conversation_id: str,
        payload: UpdateConversationRequest,
        user: User,
    ) -> ConversationResponse:
        conversation = await self._get_owned(conversation_id, user)
        conversation.title = payload.title
        await self.session.commit()
        await self.session.refresh(conversation)
        return ConversationResponse.model_validate(conversation)

    async def delete_conversation(
        self,
        conversation_id: str,
        user: User,
    ) -> None:
        conversation = await self._get_owned(conversation_id, user)
        await self.session.delete(conversation)
        await self.session.commit()

    async def list_messages(
        self,
        conversation_id: str,
        user: User,
    ) -> ConversationMessagesResponse:
        # Verify ownership first
        await self._get_owned(conversation_id, user)

        result = await self.session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        messages = result.scalars().all()
        return ConversationMessagesResponse(
            messages=[MessageResponse.model_validate(m) for m in messages],
            total=len(messages),
        )

    async def _get_owned(
        self,
        conversation_id: str,
        user: User,
    ) -> Conversation:
        result = await self.session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user.id,
            )
        )
        conversation = result.scalar_one_or_none()
        if conversation is None:
            from fastapi import HTTPException, status

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )
        return conversation
