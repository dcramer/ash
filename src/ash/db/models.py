"""SQLAlchemy ORM models.

Note: Session and Message models have been removed in favor of JSONL storage.
See ash.sessions module for the new session management system.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Base class for all models."""

    type_annotation_map = {
        dict[str, Any]: JSON,
    }


class Person(Base):
    """Person entity that memories can be about.

    Tracks people the user mentions (wife, boss, friends, etc.) so that
    memories can be properly attributed and retrieved.
    """

    __tablename__ = "people"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    owner_user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    relation: Mapped[str | None] = mapped_column(String, nullable=True)
    aliases: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSON, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now, onupdate=utc_now, nullable=False
    )


class Memory(Base):
    """Memory entry - a stored fact or piece of information.

    Memory scoping:
    - Personal: owner_user_id set, chat_id NULL - only visible to that user
    - Group: owner_user_id NULL, chat_id set - visible to everyone in that chat
    - Global: both NULL - visible everywhere (rare)

    Supersession:
    - When a new memory conflicts with an old one, the old one is marked superseded
    - Superseded memories are preserved for history but excluded from retrieval
    """

    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now, nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSON, nullable=True
    )

    # Owner tracking - who added this fact (NULL for group/shared memories)
    owner_user_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    # Chat/group scoping - which chat this memory belongs to (NULL for personal memories)
    chat_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    # Subject tracking - who/what is this fact about (list of person IDs)
    subject_person_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    # Supersession tracking - soft delete with history
    superseded_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, index=True
    )
    superseded_by_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("memories.id", ondelete="SET NULL"), nullable=True
    )

    superseded_by: Mapped["Memory | None"] = relationship(
        "Memory", remote_side="Memory.id", foreign_keys=[superseded_by_id]
    )


class UserProfile(Base):
    """User profile information."""

    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    username: Mapped[str | None] = mapped_column(String, nullable=True)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    profile_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now, onupdate=utc_now, nullable=False
    )


class SkillState(Base):
    """Persistent state storage for skills.

    Skills can store key-value pairs that persist across invocations.
    State can be global (user_id=None) or per-user.
    """

    __tablename__ = "skill_state"

    skill_name: Mapped[str] = mapped_column(String, primary_key=True)
    key: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str | None] = mapped_column(
        String, primary_key=True, nullable=False, default=""
    )
    value: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now, onupdate=utc_now, nullable=False
    )
