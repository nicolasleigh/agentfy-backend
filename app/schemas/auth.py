from typing import Annotated

from pydantic import BaseModel, EmailStr, Field

from app.schemas.user import UserPublic


class RegisterRequest(BaseModel):
    email: EmailStr
    password: Annotated[str, Field(min_length=8, max_length=128)]
    name: Annotated[str | None, Field(default=None, max_length=128)]


class LoginRequest(BaseModel):
    email: EmailStr
    password: Annotated[str, Field(min_length=1, max_length=128)]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic
