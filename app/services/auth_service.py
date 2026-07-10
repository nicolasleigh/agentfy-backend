from datetime import UTC, datetime
from uuid import uuid4

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse
from app.schemas.user import UserPublic


class AuthService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def register(self, payload: RegisterRequest) -> TokenResponse:
        email = payload.email.lower()
        existing_user = await self.session.scalar(select(User).where(User.email == email))
        if existing_user is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email is already registered",
            )

        user = User(
            id=f"user-{uuid4().hex}",
            email=email,
            name=payload.name,
            hashed_password=hash_password(payload.password),
            created_at=datetime.now(UTC),
        )
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)

        return self._build_token_response(user)

    async def login(self, payload: LoginRequest) -> TokenResponse:
        email = payload.email.lower()
        user = await self.session.scalar(select(User).where(User.email == email))
        if user is None or not verify_password(payload.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is inactive",
            )

        return self._build_token_response(user)

    def _build_token_response(self, user: User) -> TokenResponse:
        return TokenResponse(
            access_token=create_access_token(subject=user.id),
            user=UserPublic.model_validate(user),
        )
