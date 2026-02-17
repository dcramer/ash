"""Memory eviction, compaction, and index operations."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)


class MemoryEvictionMixin:
    """Memory eviction, compaction, clearing, and index operations."""

    async def enforce_max_entries(self: Store, max_entries: int) -> int:
        now = datetime.now(UTC)

        # Count active memories
        active = [m for m in self._graph.memories.values() if m.is_active(now)]

        if len(active) <= max_entries:
            return 0

        excess = len(active) - max_entries
        seven_days_ago = now - timedelta(days=7)

        # Get oldest memories (older than 7 days)
        candidates = sorted(
            [m for m in active if m.created_at and m.created_at < seven_days_ago],
            key=lambda m: m.created_at or datetime.min.replace(tzinfo=UTC),
        )
        ids_to_evict = [m.id for m in candidates[:excess]]

        for mid in ids_to_evict:
            memory = self._graph.memories.get(mid)
            if memory:
                memory.archived_at = now
                memory.archive_reason = "evicted"

        if ids_to_evict:
            self._persistence.mark_dirty("memories")
            await self._persistence.flush(self._graph)

        await self._remove_from_vector_index(ids_to_evict)

        if len(ids_to_evict) < excess:
            logger.warning(
                "Could not evict enough memories - all remaining are recent",
                extra={"excess": excess, "evicted": len(ids_to_evict)},
            )

        return len(ids_to_evict)

    async def compact(self: Store, older_than_days: int = 90) -> int:
        """Permanently remove old archived entries."""
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)

        ids_to_remove: list[str] = []
        for memory in list(self._graph.memories.values()):
            if memory.archived_at and memory.archived_at < cutoff:
                ids_to_remove.append(memory.id)

        for mid in ids_to_remove:
            self._graph.remove_memory(mid)

        if ids_to_remove:
            self._persistence.mark_dirty("memories", "edges")
            await self._persistence.flush(self._graph)
            logger.info("compact_complete", extra={"removed_count": len(ids_to_remove)})

        return len(ids_to_remove)

    async def clear(self: Store) -> int:
        """Clear all memories and vector index."""
        memory_ids = list(self._graph.memories.keys())
        count = len(memory_ids)

        if count > 0:
            for mid in memory_ids:
                self._graph.remove_memory(mid)
            self._persistence.mark_dirty("memories", "edges")
            await self._persistence.flush(self._graph)

        self._index.clear()

        await self._save_vector_index()

        return count

    async def rebuild_index(self: Store) -> int:
        """Rebuild vector index, generating embeddings for any memories missing them."""
        active_memories = [m for m in self._graph.memories.values() if m.is_active()]

        existing_ids = self._index.get_ids()

        generated = 0
        for memory in active_memories:
            if memory.id in existing_ids:
                continue
            try:
                floats = await self._embeddings.embed(memory.content)
                if floats:
                    self._index.add(memory.id, floats)
                    generated += 1
            except Exception:
                logger.debug(
                    "Failed to generate embedding for %s during rebuild", memory.id
                )

        if generated:
            await self._save_vector_index()
            logger.info(
                "Generated embeddings during rebuild",
                extra={"generated": generated, "already_indexed": len(existing_ids)},
            )

        return generated

    async def remap_subject_person_id(self: Store, old_id: str, new_id: str) -> int:
        count = self._remap_subject_person_id_batched(old_id, new_id)
        if count > 0:
            await self._persistence.flush(self._graph)
        return count

    def _remap_subject_person_id_batched(self: Store, old_id: str, new_id: str) -> int:
        """Remap ABOUT edges from old_id to new_id. Marks dirty; caller must flush."""
        from ash.graph.edges import ABOUT, create_about_edge, get_memories_about_person

        memory_ids = get_memories_about_person(self._graph, old_id)
        count = 0
        edges_changed = False
        for mid in memory_ids:
            memory = self._graph.memories.get(mid)
            if not memory:
                continue

            count += 1

            # Update ABOUT edges: remove old, add new
            old_edges = [
                e
                for e in self._graph.get_outgoing(mid, edge_type=ABOUT)
                if e.target_id == old_id
            ]
            for edge in old_edges:
                self._graph.remove_edge(edge.id)
                edges_changed = True
            # Add new edge if not already present
            existing_new = [
                e
                for e in self._graph.get_outgoing(mid, edge_type=ABOUT)
                if e.target_id == new_id
            ]
            if not existing_new:
                self._graph.add_edge(create_about_edge(mid, new_id, created_by="remap"))
                edges_changed = True

        if count > 0:
            if edges_changed:
                self._persistence.mark_dirty("edges")
            logger.debug(
                "remapped_subject_person_id",
                extra={"old_id": old_id, "new_id": new_id, "count": count},
            )

        return count
