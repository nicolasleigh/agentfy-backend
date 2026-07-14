import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, TypeDecorator, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.conversation import Conversation
    from app.models.document import Document


class VectorType(TypeDecorator):
    """A cross-dialect float-vector column.

    - **PostgreSQL (pgvector)**: stores as ``vector(768)``.
    - **SQLite / others**: stores as a JSON text string.
    """

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector
            return dialect.type_descriptor(Vector(768))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value: list[float] | None, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return value  # pgvector handles list[float] natively
        if value is not None:
            return json.dumps(value)
        return None

    def process_result_value(self, value: Any | None, dialect: Any) -> list[float] | None:
        if dialect.name == "postgresql":
            return value  # pgvector returns list[float] natively
        if isinstance(value, str):
            return json.loads(value)
        return value


class DocumentChunk(Base):
    """A chunk of text from a document, with its vector embedding."""

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        VectorType, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="chunks")
