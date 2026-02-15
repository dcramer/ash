"""Row mappers for converting database rows to domain types.

Centralizes row-to-object conversion logic, making it reusable and testable.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ash.store.types import (
    AliasEntry,
    ChatEntry,
    MemoryEntry,
    MemoryType,
    PersonEntry,
    RelationshipClaim,
    Sensitivity,
    UserEntry,
    _parse_datetime,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Row


def row_to_memory(row: Row[Any]) -> MemoryEntry:
    """Convert a SQLite row to a MemoryEntry.

    Note: subject_person_ids is set to an empty list. Use load_memory_subjects
    or row_to_memory_full to populate it.
    """
    return MemoryEntry(
        id=row.id,
        version=row.version,
        content=row.content,
        memory_type=MemoryType(row.memory_type),
        created_at=_parse_datetime(row.created_at),
        observed_at=_parse_datetime(row.observed_at),
        owner_user_id=row.owner_user_id,
        chat_id=row.chat_id,
        subject_person_ids=[],  # Loaded separately from memory_subjects
        source=row.source,
        source_username=row.source_username,
        source_display_name=row.source_display_name,
        source_session_id=row.source_session_id,
        source_message_id=row.source_message_id,
        extraction_confidence=row.extraction_confidence,
        sensitivity=Sensitivity(row.sensitivity) if row.sensitivity else None,
        portable=bool(row.portable),
        expires_at=_parse_datetime(row.expires_at),
        superseded_at=_parse_datetime(row.superseded_at),
        superseded_by_id=row.superseded_by_id,
        archived_at=_parse_datetime(row.archived_at),
        archive_reason=row.archive_reason,
        metadata=json.loads(row.metadata) if row.metadata else None,
    )


def row_to_person(
    row: Row[Any], aliases: list[AliasEntry], relationships: list[RelationshipClaim]
) -> PersonEntry:
    """Convert a SQLite row + loaded sub-records to a PersonEntry."""
    return PersonEntry(
        id=row.id,
        version=row.version,
        created_by=row.created_by,
        name=row.name,
        relationships=relationships,
        aliases=aliases,
        merged_into=row.merged_into,
        created_at=_parse_datetime(row.created_at),
        updated_at=_parse_datetime(row.updated_at),
        metadata=json.loads(row.metadata) if row.metadata else None,
    )


def row_to_user(row: Row[Any]) -> UserEntry:
    """Convert a SQLite row to a UserEntry."""
    return UserEntry(
        id=row.id,
        version=row.version,
        provider=row.provider,
        provider_id=row.provider_id,
        username=row.username,
        display_name=row.display_name,
        person_id=row.person_id,
        created_at=_parse_datetime(row.created_at),
        updated_at=_parse_datetime(row.updated_at),
        metadata=json.loads(row.metadata) if row.metadata else None,
    )


def row_to_chat(row: Row[Any]) -> ChatEntry:
    """Convert a SQLite row to a ChatEntry."""
    return ChatEntry(
        id=row.id,
        version=row.version,
        provider=row.provider,
        provider_id=row.provider_id,
        chat_type=row.chat_type,
        title=row.title,
        created_at=_parse_datetime(row.created_at),
        updated_at=_parse_datetime(row.updated_at),
        metadata=json.loads(row.metadata) if row.metadata else None,
    )


def row_to_alias(row: Row[Any]) -> AliasEntry:
    """Convert a SQLite row to an AliasEntry.

    Handles both tuple rows (from index-based queries) and named rows.
    """
    return AliasEntry(
        value=row[0] if isinstance(row, tuple) else row.value,
        added_by=row[1] if isinstance(row, tuple) else row.added_by,
        created_at=_parse_datetime(
            row[2] if isinstance(row, tuple) else row.created_at
        ),
    )


def row_to_relationship(row: Row[Any]) -> RelationshipClaim:
    """Convert a SQLite row to a RelationshipClaim.

    Handles both tuple rows (from index-based queries) and named rows.
    """
    return RelationshipClaim(
        relationship=row[0] if isinstance(row, tuple) else row.relationship,
        stated_by=row[1] if isinstance(row, tuple) else row.stated_by,
        created_at=_parse_datetime(
            row[2] if isinstance(row, tuple) else row.created_at
        ),
    )


def parse_dt(val: str | datetime | None) -> datetime | None:
    """Parse ISO datetime string from SQLite.

    Handles both string and datetime inputs for flexibility.
    A convenience wrapper around _parse_datetime that also accepts
    datetime objects passthrough.
    """
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    return _parse_datetime(val)
