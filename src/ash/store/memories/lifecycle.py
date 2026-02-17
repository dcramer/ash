"""Memory lifecycle operations: GC, expiration, archival."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

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

        archived_ids: list[str] = []
        for memory in self._graph.memories.values():
            if memory.archived_at is not None:
                continue
            reason = self._should_archive(memory, now)
            if reason:
                memory.archived_at = now
                memory.archive_reason = reason
                archived_ids.append(memory.id)
                logger.info(
                    "gc_archive_memory",
                    extra={"memory_id": memory.id, "reason": reason},
                )

        if archived_ids:
            self._persistence.mark_dirty("memories")
            await self._persistence.flush(self._graph)
            await self._remove_from_vector_index(archived_ids)

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
        from ash.graph.edges import get_memories_about_person

        now = datetime.now(UTC)

        to_archive: list[str] = []
        for mid in get_memories_about_person(self._graph, person_id):
            memory = self._graph.memories.get(mid)
            if memory and memory.archived_at is None:
                to_archive.append(mid)

        if not to_archive:
            if delete_person_record:
                await self.delete_person(person_id)
            return 0

        for mid in to_archive:
            memory = self._graph.memories.get(mid)
            if memory:
                memory.archived_at = now
                memory.archive_reason = "forgotten"

        self._persistence.mark_dirty("memories")
        await self._persistence.flush(self._graph)

        await self._remove_from_vector_index(to_archive)

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
        """Build chain: find all memories that were superseded leading to this one."""
        from ash.graph.edges import get_supersession_targets

        result: list[MemoryEntry] = []
        current_id = memory_id

        for _ in range(100):  # Safety limit
            target_ids = get_supersession_targets(self._graph, current_id)
            predecessors = [
                self._graph.memories[mid]
                for mid in target_ids
                if mid in self._graph.memories
            ]
            if not predecessors:
                break
            result.extend(predecessors)
            current_id = predecessors[0].id

        result.reverse()
        return result

    async def archive_memories(
        self: Store, memory_ids: set[str], reason: str
    ) -> list[str]:
        """Archive memories and clean up vector index."""
        if not memory_ids:
            return []

        now = datetime.now(UTC)
        archived: list[str] = []

        for mid in memory_ids:
            memory = self._graph.memories.get(mid)
            if memory and memory.archived_at is None:
                memory.archived_at = now
                memory.archive_reason = reason
                archived.append(mid)

        if archived:
            self._persistence.mark_dirty("memories")
            await self._persistence.flush(self._graph)

        await self._remove_from_vector_index(archived)

        return archived

    async def memories_about_person(self: Store, person_id: str) -> set[str]:
        """Return active memory IDs that reference a person."""
        from ash.graph.edges import get_memories_about_person

        result: set[str] = set()
        for mid in get_memories_about_person(self._graph, person_id):
            memory = self._graph.memories.get(mid)
            if memory and memory.archived_at is None:
                result.add(mid)
        return result
