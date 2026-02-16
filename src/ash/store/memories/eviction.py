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
        active = [
            m
            for m in self._graph.memories.values()
            if m.archived_at is None
            and m.superseded_at is None
            and (m.expires_at is None or m.expires_at > now)
        ]

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
            await self._persistence.save_memories(self._graph.memories)

        for mid in ids_to_evict:
            try:
                self._index.remove(mid)
            except Exception:
                logger.warning("Failed to delete embedding for %s during eviction", mid)

        if ids_to_evict:
            try:
                await self._index.save(
                    self._persistence.graph_dir / "embeddings" / "memories.npy"
                )
            except Exception:
                logger.debug("Failed to save index after eviction")

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
            await self._persistence.save_memories(self._graph.memories)
            logger.info("compact_complete", extra={"removed_count": len(ids_to_remove)})

        return len(ids_to_remove)

    async def clear(self: Store) -> int:
        """Clear all memories and vector index."""
        count = len(self._graph.memories)

        if count > 0:
            self._graph.memories.clear()
            await self._persistence.save_memories(self._graph.memories)

        # Clear vector index
        self._index._ids.clear()
        self._index._id_to_index.clear()
        import numpy as np

        self._index._vectors = np.empty((0, 0), dtype=np.float32)

        try:
            await self._index.save(
                self._persistence.graph_dir / "embeddings" / "memories.npy"
            )
        except Exception:
            logger.debug("Failed to save cleared index", exc_info=True)

        return count

    async def rebuild_index(self: Store) -> int:
        """Rebuild vector index, generating embeddings for any memories missing them."""
        active_memories = [
            m
            for m in self._graph.memories.values()
            if m.archived_at is None and m.superseded_at is None
        ]

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
            await self._index.save(
                self._persistence.graph_dir / "embeddings" / "memories.npy"
            )
            logger.info(
                "Generated embeddings during rebuild",
                extra={"generated": generated, "already_indexed": len(existing_ids)},
            )

        return generated

    async def remap_subject_person_id(self: Store, old_id: str, new_id: str) -> int:
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
                await self._persistence.save_edges(self._graph.edges)
            logger.debug(
                "remapped_subject_person_id",
                extra={"old_id": old_id, "new_id": new_id, "count": count},
            )

        return count
