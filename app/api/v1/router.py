from fastapi import APIRouter

from app.api.v1 import auth, chat, conversation

api_router = APIRouter()
api_router.include_router(auth.router, tags=["auth"])
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(conversation.router, tags=["conversations"])
