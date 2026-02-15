"""SQLAlchemy ORM models.

These models define the database schema for test setup (Base.metadata.create_all)
and Alembic migrations. The production code uses raw SQL via sqlalchemy.text().

The graph tables (memories, people, users, chats, and their join tables) are the
primary schema. The memory_embeddings virtual table is created by VectorIndex.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Base class for all models."""

    type_annotation_map = {
        dict[str, Any]: JSON,
    }


class Memory(Base):
    """Memory entry stored in the graph."""

    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    memory_type: Mapped[str] = mapped_column(
        String, nullable=False, default="knowledge"
    )
    source: Mapped[str] = mapped_column(String, nullable=False, default="user")
    owner_user_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    chat_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    source_username: Mapped[str | None] = mapped_column(String, nullable=True)
    source_display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    source_session_id: Mapped[str | None] = mapped_column(String, nullable=True)
    source_message_id: Mapped[str | None] = mapped_column(String, nullable=True)
    extraction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    sensitivity: Mapped[str | None] = mapped_column(String, nullable=True)
    portable: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    observed_at: Mapped[str | None] = mapped_column(String, nullable=True)
    expires_at: Mapped[str | None] = mapped_column(String, nullable=True)
    superseded_at: Mapped[str | None] = mapped_column(String, nullable=True)
    superseded_by_id: Mapped[str | None] = mapped_column(String, nullable=True)
    archived_at: Mapped[str | None] = mapped_column(String, nullable=True)
    archive_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_: Mapped[str | None] = mapped_column("metadata", Text, nullable=True)


class MemorySubject(Base):
    """Join table: memory -> person (subject)."""

    __tablename__ = "memory_subjects"

    memory_id: Mapped[str] = mapped_column(String, primary_key=True)
    person_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)


class Person(Base):
    """Person entity in the graph."""

    __tablename__ = "people"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String, nullable=False, default="")
    name: Mapped[str] = mapped_column(String, nullable=False, default="")
    merged_into: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)
    metadata_: Mapped[str | None] = mapped_column("metadata", Text, nullable=True)


class PersonAlias(Base):
    """Alias for a person."""

    __tablename__ = "person_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    person_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    value: Mapped[str] = mapped_column(String, nullable=False)
    added_by: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str | None] = mapped_column(String, nullable=True)


class PersonRelationship(Base):
    """Relationship claim for a person."""

    __tablename__ = "person_relationships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    person_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    relationship: Mapped[str] = mapped_column(String, nullable=False)
    stated_by: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str | None] = mapped_column(String, nullable=True)


class User(Base):
    """Provider user identity."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    provider: Mapped[str] = mapped_column(String, nullable=False, default="")
    provider_id: Mapped[str] = mapped_column(String, nullable=False, default="")
    username: Mapped[str | None] = mapped_column(String, nullable=True)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    person_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)
    metadata_: Mapped[str | None] = mapped_column("metadata", Text, nullable=True)


class Chat(Base):
    """Chat/channel identity."""

    __tablename__ = "chats"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    provider: Mapped[str] = mapped_column(String, nullable=False, default="")
    provider_id: Mapped[str] = mapped_column(String, nullable=False, default="")
    chat_type: Mapped[str | None] = mapped_column(String, nullable=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)
    metadata_: Mapped[str | None] = mapped_column("metadata", Text, nullable=True)
