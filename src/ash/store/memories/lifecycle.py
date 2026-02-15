"""Memory lifecycle operations: GC, expiration, archival."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import text

from ash.store.mappers import row_to_memory as _row_to_memory
from ash.store.memories.helpers import load_subjects
from ash.store.types import (
    EPHEMERAL_TYPES,
    TYPE_TTL,
    GCResult,
    MemoryEntry,
)

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)


class MemoryLifecycleMixin:
    """Memory garbage collection, expiration, and archival operations."""

    async def gc(self: Store) -> GCResult:
        now = datetime.now(UTC)
        now_iso = now.isoformat()

        async with self._db.session() as session:
            # Find active memories that should be archived
            result = await session.execute(
                text("SELECT * FROM memories WHERE archived_at IS NULL"),
            )
            rows = result.fetchall()

            archived_ids: list[str] = []
            for row in rows:
                memory = _row_to_memory(row)
                reason = self._should_archive(memory, now)
                if reason:
                    await session.execute(
                        text(
                            "UPDATE memories SET archived_at = :now, archive_reason = :reason WHERE id = :id"
                        ),
                        {"now": now_iso, "reason": reason, "id": memory.id},
                    )
                    archived_ids.append(memory.id)
                    logger.info(
                        "gc_archive_memory",
                        extra={"memory_id": memory.id, "reason": reason},
                    )

        for memory_id in archived_ids:
            try:
                await self._index.delete_embedding(memory_id)
            except Exception:
                logger.debug("Failed to delete embedding for %s", memory_id)

        logger.info("gc_complete", extra={"removed_count": len(archived_ids)})
        return GCResult(removed_count=len(archived_ids), archived_ids=archived_ids)

    @staticmethod
    def _should_archive(memory: MemoryEntry, now: datetime) -> str | None:
        """Return archive reason if memory should be archived, else None."""
        if memory.superseded_at:
            return "superseded"
        if memory.expires_at and memory.expires_at <= now:
            return "expired"
        if not memory.expires_at and memory.memory_type in EPHEMERAL_TYPES:
            anchor = memory.observed_at or memory.created_at
            if anchor:
                age_days = (now - anchor).days
                default_ttl = TYPE_TTL.get(memory.memory_type, 30)
                if age_days > default_ttl:
                    return "ephemeral_decay"
        return None

    async def forget_person(
        self: Store,
        person_id: str,
        delete_person_record: bool = False,
    ) -> int:
        now_iso = datetime.now(UTC).isoformat()

        async with self._db.session() as session:
            # Find memories about this person that are active
            result = await session.execute(
                text("""
                    SELECT ms.memory_id FROM memory_subjects ms
                    JOIN memories m ON m.id = ms.memory_id
                    WHERE ms.person_id = :pid AND m.archived_at IS NULL
                """),
                {"pid": person_id},
            )
            to_archive = [row[0] for row in result.fetchall()]

            if not to_archive:
                if delete_person_record:
                    await self.delete_person(person_id)
                return 0

            for mid in to_archive:
                await session.execute(
                    text(
                        "UPDATE memories SET archived_at = :now, archive_reason = 'forgotten' WHERE id = :id AND archived_at IS NULL"
                    ),
                    {"now": now_iso, "id": mid},
                )

        for memory_id in to_archive:
            try:
                await self._index.delete_embedding(memory_id)
            except Exception:
                logger.debug("Failed to delete embedding for %s", memory_id)

        if delete_person_record:
            await self.delete_person(person_id)

        logger.info(
            "forget_person_complete",
            extra={
                "person_id": person_id,
                "archived_count": len(to_archive),
                "deleted_person": delete_person_record,
            },
        )
        return len(to_archive)

    async def get_supersession_chain(self: Store, memory_id: str) -> list[MemoryEntry]:
        async with self._db.session() as session:
            # Build chain: find all memories that were superseded leading to this one
            result: list[MemoryEntry] = []
            current_id = memory_id

            # Walk backwards through superseded_by_id links
            for _ in range(100):  # Safety limit
                r = await session.execute(
                    text("SELECT * FROM memories WHERE superseded_by_id = :id"),
                    {"id": current_id},
                )
                predecessors = r.fetchall()
                if not predecessors:
                    break
                for row in predecessors:
                    memory = _row_to_memory(row)
                    memory.subject_person_ids = await load_subjects(session, memory.id)
                    result.append(memory)
                current_id = predecessors[0].id

            result.reverse()
            return result

    async def archive_memories(
        self: Store, memory_ids: set[str], reason: str
    ) -> list[str]:
        """Archive memories and clean up vector index."""
        if not memory_ids:
            return []

        now_iso = datetime.now(UTC).isoformat()
        archived: list[str] = []

        async with self._db.session() as session:
            for mid in memory_ids:
                r = await session.execute(
                    text(
                        "UPDATE memories SET archived_at = :now, archive_reason = :reason WHERE id = :id AND archived_at IS NULL"
                    ),
                    {"now": now_iso, "reason": reason, "id": mid},
                )
                if r.rowcount > 0:  # type: ignore[possibly-missing-attribute]
                    archived.append(mid)

        for memory_id in archived:
            try:
                await self._index.delete_embedding(memory_id)
            except Exception:
                logger.debug("Failed to delete embedding for %s", memory_id)

        return archived

    async def memories_about_person(self: Store, person_id: str) -> set[str]:
        """Return active memory IDs that reference a person via memory_subjects."""
        async with self._db.session() as session:
            result = await session.execute(
                text("""
                    SELECT ms.memory_id FROM memory_subjects ms
                    JOIN memories m ON m.id = ms.memory_id
                    WHERE ms.person_id = :pid AND m.archived_at IS NULL
                """),
                {"pid": person_id},
            )
            return {row[0] for row in result.fetchall()}
