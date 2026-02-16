"""Memory CRUD operations: create, read, update, delete."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ash.graph.edges import create_about_edge
from ash.memory.secrets import contains_secret
from ash.store.types import (
    MemoryEntry,
    MemoryType,
    Sensitivity,
    matches_scope,
)

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)


class MemoryCrudMixin:
    """Memory create, read, update, delete operations."""

    async def _resolve_subject_name(self: Store, person_ids: list[str]) -> str | None:
        if not person_ids:
            return None

        names: list[str] = []
        cache_misses: list[str] = []

        for pid in person_ids:
            if pid in self._person_name_cache:
                names.append(self._person_name_cache[pid])
            else:
                cache_misses.append(pid)

        if cache_misses:
            batch_names: dict[str, str] = {}
            try:
                batch_names = await self.get_person_names_batch(cache_misses)
            except Exception:
                logger.debug("Failed to batch resolve person names, falling back")

            for pid in cache_misses:
                if pid in batch_names:
                    name = batch_names[pid]
                    self._person_name_cache[pid] = name
                    names.append(name)
                else:
                    try:
                        person = await self.get_person(pid)
                        if person:
                            self._person_name_cache[pid] = person.name
                            names.append(person.name)
                    except Exception:
                        logger.debug("Failed to resolve person name for %s", pid)

        return ", ".join(names) if names else None

    async def add_memory(
        self: Store,
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

        now = datetime.now(UTC)
        memory_id = str(uuid.uuid4())
        subject_pids = subject_person_ids or []

        memory = MemoryEntry(
            id=memory_id,
            version=1,
            content=content,
            memory_type=memory_type,
            created_at=now,
            observed_at=observed_at,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            subject_person_ids=subject_pids,
            source=source,
            source_username=source_username,
            source_display_name=source_display_name,
            source_session_id=source_session_id,
            source_message_id=source_message_id,
            extraction_confidence=extraction_confidence,
            sensitivity=sensitivity,
            portable=portable,
            expires_at=expires_at,
            metadata=metadata,
        )

        # Add to graph and persist
        self._graph.add_memory(memory)
        await self._persistence.save_memories(self._graph.memories)

        # Dual-write ABOUT edges for subject_person_ids
        if subject_pids:
            for pid in subject_pids:
                edge = create_about_edge(memory_id, pid, created_by=source)
                self._graph.add_edge(edge)
            await self._persistence.save_edges(self._graph.edges)

        if embedding_floats:
            embedding_base64 = MemoryEntry.encode_embedding(embedding_floats)
            try:
                self._index.add(memory.id, embedding_floats)
                await self._index.save(
                    self._persistence.graph_dir / "embeddings" / "memories.npy"
                )
            except Exception:
                logger.warning("Failed to index memory, continuing", exc_info=True)
            memory.embedding = embedding_base64

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

        logger.debug(
            "memory_added",
            extra={
                "memory_id": memory.id,
                "memory_type": memory_type.value,
                "source": source,
            },
        )

        return memory

    async def get_memory(self: Store, memory_id: str) -> MemoryEntry | None:
        memory = self._graph.memories.get(memory_id)
        if memory and memory.archived_at is None:
            return memory
        return None

    async def get_memory_by_prefix(self: Store, prefix: str) -> MemoryEntry | None:
        """Find a memory by ID prefix match."""
        # Try exact match first
        memory = self._graph.memories.get(prefix)
        if memory and memory.archived_at is None:
            return memory

        # Try prefix match
        matches = [
            m
            for mid, m in self._graph.memories.items()
            if mid.startswith(prefix) and m.archived_at is None
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    async def list_memories(
        self: Store,
        limit: int | None = 20,
        include_expired: bool = False,
        include_superseded: bool = False,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[MemoryEntry]:
        now = datetime.now(UTC)
        results: list[MemoryEntry] = []

        for memory in self._graph.memories.values():
            if memory.archived_at is not None:
                continue
            if not include_expired:
                if memory.expires_at and memory.expires_at <= now:
                    continue
            if not include_superseded:
                if memory.superseded_at:
                    continue
            if not matches_scope(memory, owner_user_id, chat_id):
                continue
            results.append(memory)

        # Sort by created_at descending
        results.sort(
            key=lambda m: m.created_at or datetime.min.replace(tzinfo=UTC), reverse=True
        )

        if limit is not None:
            results = results[:limit]

        return results

    async def delete_memory(
        self: Store,
        memory_id: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> bool:
        memory = await self.get_memory_by_prefix(memory_id)
        if not memory:
            return False

        full_id = memory.id

        # Check authorization
        if owner_user_id or chat_id:
            is_owner = owner_user_id and memory.owner_user_id == owner_user_id
            is_group_member = memory.owner_user_id is None and memory.chat_id == chat_id
            if not (is_owner or is_group_member):
                return False

        memory.archived_at = datetime.now(UTC)
        memory.archive_reason = "user_deleted"
        await self._persistence.save_memories(self._graph.memories)

        try:
            self._index.remove(full_id)
            await self._index.save(
                self._persistence.graph_dir / "embeddings" / "memories.npy"
            )
        except Exception:
            logger.warning(
                "Failed to delete memory embedding",
                extra={"memory_id": full_id},
                exc_info=True,
            )

        logger.info("memory_deleted", extra={"memory_id": full_id})
        return True

    async def batch_update_memories(self: Store, entries: list[MemoryEntry]) -> int:
        """Update multiple memories."""
        if not entries:
            return 0

        count = 0
        edges_changed = False
        for entry in entries:
            if entry.id in self._graph.memories:
                self._graph.memories[entry.id] = entry
                count += 1

                # Sync ABOUT edges with subject_person_ids
                existing_about = self._graph.get_outgoing(entry.id, edge_type="ABOUT")
                existing_pids = {e.target_id for e in existing_about}
                desired_pids = set(entry.subject_person_ids or [])

                # Remove edges for persons no longer in subject list
                for edge in existing_about:
                    if edge.target_id not in desired_pids:
                        self._graph.remove_edge(edge.id)
                        edges_changed = True

                # Add edges for newly added persons
                for pid in desired_pids - existing_pids:
                    edge = create_about_edge(entry.id, pid, created_by=entry.source)
                    self._graph.add_edge(edge)
                    edges_changed = True

        if count > 0:
            await self._persistence.save_memories(self._graph.memories)
            if edges_changed:
                await self._persistence.save_edges(self._graph.edges)

        return count

    async def get_all_memories(self: Store) -> list[MemoryEntry]:
        """Get all memories including archived."""
        return list(self._graph.memories.values())
