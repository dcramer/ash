"""Memory eviction, compaction, and index operations."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)


class MemoryEvictionMixin:
    """Memory eviction, compaction, clearing, and index operations."""

    async def enforce_max_entries(self: Store, max_entries: int) -> int:
        now = datetime.now(UTC)
        now_iso = now.isoformat()

        async with self._db.session() as session:
            result = await session.execute(
                text(
                    "SELECT COUNT(*) FROM memories WHERE archived_at IS NULL AND superseded_at IS NULL AND (expires_at IS NULL OR expires_at > :now)"
                ),
                {"now": now_iso},
            )
            current_count = result.scalar() or 0

            if current_count <= max_entries:
                return 0

            excess = current_count - max_entries
            # Get oldest memories (older than 7 days)
            seven_days_ago = (now - timedelta(days=7)).isoformat()
            result = await session.execute(
                text("""
                    SELECT id FROM memories
                    WHERE archived_at IS NULL AND superseded_at IS NULL
                        AND (expires_at IS NULL OR expires_at > :now)
                        AND created_at < :cutoff
                    ORDER BY created_at ASC
                    LIMIT :limit
                """),
                {"now": now_iso, "cutoff": seven_days_ago, "limit": excess},
            )
            ids_to_evict = [row[0] for row in result.fetchall()]

            for mid in ids_to_evict:
                await session.execute(
                    text(
                        "UPDATE memories SET archived_at = :now, archive_reason = 'evicted' WHERE id = :id"
                    ),
                    {"now": now_iso, "id": mid},
                )

        for mid in ids_to_evict:
            try:
                await self._index.delete_embedding(mid)
            except Exception:
                logger.warning("Failed to delete embedding for %s during eviction", mid)

        if len(ids_to_evict) < excess:
            logger.warning(
                "Could not evict enough memories - all remaining are recent",
                extra={"excess": excess, "evicted": len(ids_to_evict)},
            )

        return len(ids_to_evict)

    async def compact(self: Store, older_than_days: int = 90) -> int:
        """Permanently remove old archived entries."""
        cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()

        async with self._db.session() as session:
            # Get IDs to remove (for embedding cleanup)
            result = await session.execute(
                text(
                    "SELECT id FROM memories WHERE archived_at IS NOT NULL AND archived_at < :cutoff"
                ),
                {"cutoff": cutoff},
            )
            ids_to_remove = [row[0] for row in result.fetchall()]

            if ids_to_remove:
                # Delete memory_subjects first (FK)
                for mid in ids_to_remove:
                    await session.execute(
                        text("DELETE FROM memory_subjects WHERE memory_id = :id"),
                        {"id": mid},
                    )
                # Delete memories
                for mid in ids_to_remove:
                    await session.execute(
                        text("DELETE FROM memories WHERE id = :id"),
                        {"id": mid},
                    )

                logger.info(
                    "compact_complete", extra={"removed_count": len(ids_to_remove)}
                )

        return len(ids_to_remove)

    async def clear(self: Store) -> int:
        """Clear all memories and vector index."""
        async with self._db.session() as session:
            result = await session.execute(text("SELECT COUNT(*) FROM memories"))
            count = result.scalar() or 0

            if count > 0:
                await session.execute(text("DELETE FROM memory_subjects"))
                await session.execute(text("DELETE FROM memories"))

        try:
            await self._index.clear()
        except Exception:
            logger.debug("Failed to clear vector index", exc_info=True)

        return count

    async def rebuild_index(self: Store) -> int:
        """Rebuild vector index, generating embeddings for any memories missing them."""
        async with self._db.session() as session:
            result = await session.execute(
                text(
                    "SELECT id, content FROM memories WHERE archived_at IS NULL AND superseded_at IS NULL"
                ),
            )
            active_memories = result.fetchall()

        # Only generate embeddings for memories not already indexed
        existing_ids = await self._index.get_indexed_memory_ids()

        generated = 0
        for row in active_memories:
            if row.id in existing_ids:
                continue
            try:
                floats = await self._embeddings.embed(row.content)
                if floats:
                    await self._index.add_embedding(row.id, floats)
                    generated += 1
            except Exception:
                logger.debug(
                    "Failed to generate embedding for %s during rebuild", row.id
                )

        if generated:
            logger.info(
                "Generated embeddings during rebuild",
                extra={"generated": generated, "already_indexed": len(existing_ids)},
            )

        return generated

    async def remap_subject_person_id(self: Store, old_id: str, new_id: str) -> int:
        async with self._db.session() as session:
            result = await session.execute(
                text("SELECT memory_id FROM memory_subjects WHERE person_id = :old_id"),
                {"old_id": old_id},
            )
            affected = [row[0] for row in result.fetchall()]

            if not affected:
                return 0

            # For each affected memory, check if new_id already exists
            for mid in affected:
                result = await session.execute(
                    text(
                        "SELECT COUNT(*) FROM memory_subjects WHERE memory_id = :mid AND person_id = :new_id"
                    ),
                    {"mid": mid, "new_id": new_id},
                )
                exists = (result.scalar() or 0) > 0
                if exists:
                    # Just delete the old one
                    await session.execute(
                        text(
                            "DELETE FROM memory_subjects WHERE memory_id = :mid AND person_id = :old_id"
                        ),
                        {"mid": mid, "old_id": old_id},
                    )
                else:
                    # Update old -> new
                    await session.execute(
                        text(
                            "UPDATE memory_subjects SET person_id = :new_id WHERE memory_id = :mid AND person_id = :old_id"
                        ),
                        {"mid": mid, "old_id": old_id, "new_id": new_id},
                    )

            logger.debug(
                "remapped_subject_person_id",
                extra={"old_id": old_id, "new_id": new_id, "count": len(affected)},
            )

        return len(affected)
