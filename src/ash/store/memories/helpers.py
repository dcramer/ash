"""Shared helpers for memory operations."""

from __future__ import annotations

from sqlalchemy import text

from ash.store.mappers import row_to_memory as _row_to_memory
from ash.store.types import MemoryEntry


async def load_subjects(session, memory_id: str) -> list[str]:
    """Load subject_person_ids for a memory."""
    result = await session.execute(
        text("SELECT person_id FROM memory_subjects WHERE memory_id = :id"),
        {"id": memory_id},
    )
    return [row[0] for row in result.fetchall()]


async def load_subjects_batch(session, memory_ids: list[str]) -> dict[str, list[str]]:
    """Load subject_person_ids for multiple memories."""
    if not memory_ids:
        return {}
    result = await session.execute(
        text("SELECT memory_id, person_id FROM memory_subjects"),
    )
    subjects: dict[str, list[str]] = {}
    id_set = set(memory_ids)
    for row in result.fetchall():
        if row[0] in id_set:
            subjects.setdefault(row[0], []).append(row[1])
    return subjects


async def row_to_memory_full(session, row) -> MemoryEntry:
    """Convert a row to a MemoryEntry with subject_person_ids loaded."""
    memory = _row_to_memory(row)
    memory.subject_person_ids = await load_subjects(session, memory.id)
    return memory
