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
                        logger.debug(
                            "person_name_resolve_failed",
                            extra={"person.id": pid},
                        )

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
        stated_by_person_id: str | None = None,
        graph_chat_id: str | None = None,
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
            logger.warning("embedding_generation_failed", exc_info=True)

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

        # Add to graph (in-memory)
        self._graph.add_memory(memory)
        self._persistence.mark_dirty("memories")

        # Create ABOUT edges for subject_person_ids
        if subject_pids:
            for pid in subject_pids:
                edge = create_about_edge(memory_id, pid, created_by=source)
                self._graph.add_edge(edge)
            self._persistence.mark_dirty("edges")

        # Create STATED_BY edge for speaker attribution
        if stated_by_person_id:
            from ash.graph.edges import create_stated_by_edge

            self._graph.add_edge(
                create_stated_by_edge(memory_id, stated_by_person_id, created_by=source)
            )
            self._persistence.mark_dirty("edges")

        # Create LEARNED_IN edge to track source chat
        logger.debug(
            "add_memory_graph_chat_id: memory=%s graph_chat_id=%s",
            memory_id[:8],
            graph_chat_id,
        )
        if graph_chat_id:
            from ash.graph.edges import create_learned_in_edge

            edge = create_learned_in_edge(memory_id, graph_chat_id, created_by=source)
            self._graph.add_edge(edge)
            self._persistence.mark_dirty("edges")
            logger.debug(
                "learned_in_edge_created: memory=%s chat=%s edge=%s",
                memory_id[:8],
                graph_chat_id[:8],
                edge.id[:8],
            )

        if embedding_floats:
            embedding_base64 = MemoryEntry.encode_embedding(embedding_floats)
            try:
                self._index.add(memory.id, embedding_floats)
            except Exception:
                logger.warning(
                    "memory_index_failed",
                    extra={"memory.id": memory.id},
                    exc_info=True,
                )
            memory.embedding = embedding_base64

        superseded_count = 0
        try:
            superseded_count = await self._supersede_conflicting_batched(
                new_memory=memory,
                owner_user_id=owner_user_id,
                chat_id=chat_id,
            )
            if superseded_count > 0:
                logger.info(
                    "memory_superseded",
                    extra={
                        "memory.id": memory.id,
                        "superseded_count": superseded_count,
                    },
                )
        except Exception:
            logger.warning(
                "conflicting_memories_check_failed",
                extra={"memory.id": memory.id},
                exc_info=True,
            )

        if self._max_entries is not None:
            try:
                evicted = await self.enforce_max_entries(self._max_entries)
                if evicted > 0:
                    logger.info(
                        "memories_evicted",
                        extra={"count": evicted, "max_entries": self._max_entries},
                    )
            except Exception:
                logger.warning("max_entries_enforcement_failed", exc_info=True)

        # Single flush for all mutations in this operation
        await self._persistence.flush(self._graph)

        # Save vector index after flush (separate file)
        if embedding_floats or superseded_count > 0:
            await self._save_vector_index()

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

        await self.archive_memories({full_id}, reason="user_deleted")

        logger.info("memory_deleted", extra={"memory.id": full_id})
        return True

    async def batch_update_memories(
        self: Store,
        entries: list[MemoryEntry],
        subject_person_ids_map: dict[str, list[str]] | None = None,
    ) -> int:
        """Update multiple memories.

        Args:
            entries: Memory entries to update.
            subject_person_ids_map: Optional map of memory_id -> desired subject person IDs.
                If provided, ABOUT edges will be synced for those memories.
        """
        if not entries:
            return 0

        count = 0
        edges_changed = False
        for entry in entries:
            if entry.id in self._graph.memories:
                self._graph.memories[entry.id] = entry
                count += 1

                # Sync ABOUT edges if subject_person_ids map is provided
                if subject_person_ids_map and entry.id in subject_person_ids_map:
                    from ash.graph.edges import ABOUT

                    existing_about = self._graph.get_outgoing(entry.id, edge_type=ABOUT)
                    existing_pids = {e.target_id for e in existing_about}
                    desired_pids = set(subject_person_ids_map[entry.id])

                    for edge in existing_about:
                        if edge.target_id not in desired_pids:
                            self._graph.remove_edge(edge.id)
                            edges_changed = True

                    for pid in desired_pids - existing_pids:
                        edge = create_about_edge(entry.id, pid, created_by=entry.source)
                        self._graph.add_edge(edge)
                        edges_changed = True

        if count > 0:
            self._persistence.mark_dirty("memories")
            if edges_changed:
                self._persistence.mark_dirty("edges")
            await self._persistence.flush(self._graph)

        return count

    async def get_all_memories(self: Store) -> list[MemoryEntry]:
        """Get all memories including archived."""
        return list(self._graph.memories.values())
