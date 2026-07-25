from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr


class UserPublic(BaseModel):
    # Enables UserPublic.model_validate(orm_obj) for SQLAlchemy ORM instances
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: EmailStr
    name: str | None = None
    is_active: bool
    created_at: datetime
