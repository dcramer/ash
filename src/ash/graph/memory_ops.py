"""Memory CRUD operations mixin for GraphStore."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ash.memory.secrets import contains_secret
from ash.memory.types import (
    GCResult,
    MemoryEntry,
    MemoryType,
    Sensitivity,
)

if TYPE_CHECKING:
    from ash.graph.store import GraphStore

logger = logging.getLogger(__name__)


class MemoryOpsMixin:
    """Memory CRUD, eviction, and lifecycle operations."""

    async def _resolve_subject_name(
        self: GraphStore, person_ids: list[str]
    ) -> str | None:
        if not person_ids:
            return None
        for pid in person_ids:
            if pid in self._person_name_cache:
                return self._person_name_cache[pid]
            try:
                person = await self.get_person(pid)
                if person:
                    self._person_name_cache[pid] = person.name
                    return person.name
            except Exception:
                logger.debug("Failed to resolve person name for %s", pid)
        return None

    async def add_memory(
        self: GraphStore,
        content: str,
        source: str = "user",
        memory_type: MemoryType | None = None,
        expires_at: datetime | None = None,
        expires_in_days: int | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        subject_person_ids: list[str] | None = None,
        observed_at: datetime | None = None,
        source_username: str | None = None,
        source_display_name: str | None = None,
        source_session_id: str | None = None,
        source_message_id: str | None = None,
        extraction_confidence: float | None = None,
        sensitivity: Sensitivity | None = None,
        portable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Add a memory entry."""
        if contains_secret(content):
            raise ValueError(
                "Memory content contains potential secrets and cannot be stored"
            )

        if expires_in_days is not None and expires_at is None:
            expires_at = datetime.now(UTC) + timedelta(days=expires_in_days)

        if memory_type is None:
            memory_type = MemoryType.KNOWLEDGE

        embedding_floats: list[float] | None = None
        try:
            embedding_floats = await self._embeddings.embed(content)
        except Exception:
            logger.warning(
                "Failed to generate embedding, continuing without", exc_info=True
            )

        memory = await self._store.add_memory(
            content=content,
            memory_type=memory_type,
            source=source,
            expires_at=expires_at,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            subject_person_ids=subject_person_ids,
            observed_at=observed_at,
            source_username=source_username,
            source_display_name=source_display_name,
            source_session_id=source_session_id,
            source_message_id=source_message_id,
            extraction_confidence=extraction_confidence,
            sensitivity=sensitivity,
            portable=portable,
            metadata=metadata,
        )

        if embedding_floats:
            embedding_base64 = MemoryEntry.encode_embedding(embedding_floats)
            try:
                await self._store.save_embedding(memory.id, embedding_base64)
            except Exception:
                logger.warning(
                    "Failed to save embedding to JSONL, continuing", exc_info=True
                )
            try:
                await self._index.add_embedding(memory.id, embedding_floats)
            except Exception:
                logger.warning("Failed to index memory, continuing", exc_info=True)

        try:
            superseded_count = await self.supersede_conflicting_memories(
                new_memory=memory,
                owner_user_id=owner_user_id,
                chat_id=chat_id,
            )
            if superseded_count > 0:
                logger.debug(
                    "Memory superseded %d older entries",
                    superseded_count,
                    extra={"new_memory_id": memory.id},
                )
        except Exception:
            logger.warning("Failed to check for conflicting memories", exc_info=True)

        if self._max_entries is not None:
            try:
                evicted = await self.enforce_max_entries(self._max_entries)
                if evicted > 0:
                    logger.info(
                        "Evicted memories to enforce limit",
                        extra={"evicted": evicted, "max_entries": self._max_entries},
                    )
            except Exception:
                logger.warning("Failed to enforce max_entries limit", exc_info=True)

        self._graph_built = False
        return memory

    async def get_memory(self: GraphStore, memory_id: str) -> MemoryEntry | None:
        await self._ensure_graph_built()
        memory = self._memory_by_id.get(memory_id)
        if memory and memory.archived_at is not None:
            return None
        return memory

    async def get_memory_by_prefix(self: GraphStore, prefix: str) -> MemoryEntry | None:
        """Find a memory by ID prefix match."""
        return await self._store.get_memory_by_prefix(prefix)

    async def list_memories(
        self: GraphStore,
        limit: int | None = 20,
        include_expired: bool = False,
        include_superseded: bool = False,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[MemoryEntry]:
        return await self._store.get_memories(
            include_expired=include_expired,
            include_superseded=include_superseded,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            limit=limit,
        )

    async def delete_memory(
        self: GraphStore,
        memory_id: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> bool:
        memory = await self._store.get_memory_by_prefix(memory_id)
        if not memory:
            return False

        full_id = memory.id
        deleted = await self._store.delete_memory(
            full_id, owner_user_id=owner_user_id, chat_id=chat_id
        )
        if not deleted:
            return False

        try:
            await self._index.delete_embedding(full_id)
        except Exception:
            logger.warning(
                "Failed to delete memory embedding",
                extra={"memory_id": full_id},
                exc_info=True,
            )

        self._graph_built = False
        return True

    async def gc(self: GraphStore) -> GCResult:
        result = await self._store.gc()
        for memory_id in result.archived_ids:
            try:
                await self._index.delete_embedding(memory_id)
            except Exception:
                logger.debug("Failed to delete embedding for %s", memory_id)
        self._graph_built = False
        return result

    async def compact(self: GraphStore, older_than_days: int = 90) -> int:
        """Permanently remove old archived entries and clean up vector index."""
        removed = await self._store.compact(older_than_days)
        if removed > 0:
            self._graph_built = False
        return removed

    async def clear(self: GraphStore) -> int:
        """Clear all memories, embeddings JSONL, and vector index."""
        count = await self._store.clear()
        try:
            await self._index.clear()
        except Exception:
            logger.debug("Failed to clear vector index", exc_info=True)
        self._graph_built = False
        return count

    async def enforce_max_entries(self: GraphStore, max_entries: int) -> int:
        now = datetime.now(UTC)
        memories = await self._store.get_memories(
            limit=None,
            include_expired=False,
            include_superseded=False,
        )

        current_count = len(memories)
        if current_count <= max_entries:
            return 0

        excess = current_count - max_entries
        memories.sort(key=lambda m: m.created_at or datetime.min.replace(tzinfo=UTC))

        evicted = 0
        for memory in memories:
            if evicted >= excess:
                break
            if memory.created_at and (now - memory.created_at).days < 7:
                continue
            await self._store.delete_memory(memory.id)
            try:
                await self._index.delete_embedding(memory.id)
            except Exception:
                logger.warning(
                    "Failed to delete embedding for %s during eviction", memory.id
                )
            evicted += 1

        if evicted < excess:
            logger.warning(
                "Could not evict enough memories - all remaining are recent",
                extra={"excess": excess, "evicted": evicted},
            )

        return evicted

    async def rebuild_index(self: GraphStore) -> int:
        memories = await self._store.get_all_memories()
        embeddings = await self._store.load_embeddings()
        count = await self._index.rebuild_from_embeddings(memories, embeddings)
        logger.info("index_rebuilt", extra={"count": count})
        return count

    async def remap_subject_person_id(
        self: GraphStore, old_id: str, new_id: str
    ) -> int:
        count = await self._store.remap_subject_person_id(old_id, new_id)
        if count > 0:
            self._graph_built = False
        return count

    async def forget_person(
        self: GraphStore,
        person_id: str,
        delete_person_record: bool = False,
    ) -> int:
        graph = await self._ensure_graph_built()
        candidate_ids = graph.memories_about(person_id)
        to_archive = {
            mid
            for mid in candidate_ids
            if (m := self._memory_by_id.get(mid)) and m.archived_at is None
        }

        if not to_archive:
            if delete_person_record:
                await self.delete_person(person_id)
            return 0

        archived_ids = await self._store.archive_memories(to_archive, "forgotten")
        for memory_id in archived_ids:
            try:
                await self._index.delete_embedding(memory_id)
            except Exception:
                logger.debug("Failed to delete embedding for %s", memory_id)

        if delete_person_record:
            await self.delete_person(person_id)

        self._graph_built = False
        logger.info(
            "forget_person_complete",
            extra={
                "person_id": person_id,
                "archived_count": len(archived_ids),
                "deleted_person": delete_person_record,
            },
        )
        return len(archived_ids)

    async def get_supersession_chain(
        self: GraphStore, memory_id: str
    ) -> list[MemoryEntry]:
        return await self._store.get_supersession_chain(memory_id)

    async def batch_update_memories(
        self: GraphStore, entries: list[MemoryEntry]
    ) -> int:
        """Update multiple memories and invalidate graph."""
        count = await self._store.batch_update_memories(entries)
        if count > 0:
            self._graph_built = False
        return count

    async def archive_memories(
        self: GraphStore, memory_ids: set[str], reason: str
    ) -> list[str]:
        """Archive memories and clean up vector index."""
        archived_ids = await self._store.archive_memories(memory_ids, reason)
        for memory_id in archived_ids:
            try:
                await self._index.delete_embedding(memory_id)
            except Exception:
                logger.debug("Failed to delete embedding for %s", memory_id)
        if archived_ids:
            self._graph_built = False
        return archived_ids
