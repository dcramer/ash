"""Filesystem-based memory store.

Primary storage for memories, with in-memory caching
and mtime-based invalidation.  All memories (active + archived) live
in a single JSONL file; archived entries have ``archived_at`` set.
Embeddings are stored in a separate JSONL to keep the main file
human-readable.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ash.config.paths import (
    get_embeddings_jsonl_path,
    get_memories_jsonl_path,
)
from ash.memory.jsonl import EmbeddingJSONL, MemoryJSONL
from ash.memory.types import (
    EPHEMERAL_TYPES,
    TYPE_TTL,
    EmbeddingRecord,
    GCResult,
    MemoryEntry,
    MemoryType,
    Sensitivity,
    matches_scope,
)

logger = logging.getLogger(__name__)


class FileMemoryStore:
    """Filesystem-based store for memories.

    Uses a single JSONL file as the source of truth with in-memory caching
    for fast reads.  Archived entries stay in the same file (``archived_at``
    is non-null) so supersession chains are always queryable.
    Cache is invalidated based on file mtime.
    """

    def __init__(
        self,
        memories_path: Path | None = None,
        embeddings_path: Path | None = None,
    ) -> None:
        self._memories_jsonl = MemoryJSONL(memories_path or get_memories_jsonl_path())
        self._embeddings_jsonl = EmbeddingJSONL(
            embeddings_path or get_embeddings_jsonl_path()
        )

        # In-memory cache
        self._memories_cache: list[MemoryEntry] | None = None
        self._memories_mtime: float | None = None

        # Concurrency protection for read-modify-write cycles
        import asyncio

        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    async def _ensure_memories_loaded(self) -> list[MemoryEntry]:
        """Load memories from disk if cache is stale."""
        current_mtime = self._memories_jsonl.get_mtime()

        if self._memories_cache is None or current_mtime != self._memories_mtime:
            self._memories_cache = await self._memories_jsonl.load_all()
            self._memories_mtime = current_mtime

        return self._memories_cache

    def _invalidate_memories_cache(self) -> None:
        """Invalidate memories cache after write."""
        self._memories_cache = None
        self._memories_mtime = None

    @staticmethod
    def _is_active(m: MemoryEntry) -> bool:
        """Return True if the memory has not been archived."""
        return m.archived_at is None

    # ------------------------------------------------------------------
    # Memory CRUD operations
    # ------------------------------------------------------------------

    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.KNOWLEDGE,
        source: str = "user",
        expires_at: datetime | None = None,
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
        """Add a new memory entry."""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            version=1,
            content=content,
            memory_type=memory_type,
            created_at=datetime.now(UTC),
            observed_at=observed_at,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            subject_person_ids=subject_person_ids or [],
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

        await self._memories_jsonl.append(entry)
        self._invalidate_memories_cache()

        logger.debug(
            "memory_added",
            extra={
                "memory_id": entry.id,
                "memory_type": memory_type.value,
                "source": source,
            },
        )

        return entry

    async def get_memory(self, memory_id: str) -> MemoryEntry | None:
        """Get an active memory by ID (excludes archived)."""
        memories = await self._ensure_memories_loaded()
        for m in memories:
            if m.id == memory_id and self._is_active(m):
                return m
        return None

    async def get_memory_by_prefix(self, memory_id_prefix: str) -> MemoryEntry | None:
        """Get an active memory by ID or ID prefix.

        Supports short IDs like git - if exactly one active memory matches
        the prefix, return it. Returns None if no match or multiple matches.
        """
        memories = await self._ensure_memories_loaded()
        active = [m for m in memories if self._is_active(m)]

        # Try exact match first
        for m in active:
            if m.id == memory_id_prefix:
                return m

        # Try prefix match
        matches = [m for m in active if m.id.startswith(memory_id_prefix)]
        if len(matches) == 1:
            return matches[0]
        return None

    async def get_memories(
        self,
        limit: int | None = 100,
        include_expired: bool = False,
        include_superseded: bool = False,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[MemoryEntry]:
        """Get active memory entries with filters."""
        memories = await self._ensure_memories_loaded()
        now = datetime.now(UTC)

        result: list[MemoryEntry] = []
        for m in memories:
            if not self._is_active(m):
                continue
            if not include_expired and m.expires_at and m.expires_at <= now:
                continue
            if not include_superseded and m.superseded_at:
                continue
            if not matches_scope(m, owner_user_id, chat_id):
                continue
            result.append(m)

        result.sort(
            key=lambda x: x.created_at or datetime.min.replace(tzinfo=UTC), reverse=True
        )

        if limit is not None:
            return result[:limit]
        return result

    async def mark_memory_superseded(
        self,
        memory_id: str,
        superseded_by_id: str,
    ) -> bool:
        """Mark a memory as superseded by another memory."""
        async with self._write_lock:
            memories = await self._ensure_memories_loaded()

            found = False
            for m in memories:
                if m.id == memory_id:
                    m.superseded_at = datetime.now(UTC)
                    m.superseded_by_id = superseded_by_id
                    found = True
                    break

            if found:
                await self._memories_jsonl.rewrite(memories)
                self._invalidate_memories_cache()
                logger.info(
                    "memory_superseded",
                    extra={"memory_id": memory_id, "superseded_by": superseded_by_id},
                )

        return found

    async def batch_mark_superseded(self, pairs: list[tuple[str, str]]) -> list[str]:
        """Mark multiple memories as superseded in a single rewrite.

        Args:
            pairs: List of (old_memory_id, new_memory_id) tuples.

        Returns:
            List of old memory IDs that were actually marked.
        """
        if not pairs:
            return []

        async with self._write_lock:
            supersede_map = {old_id: new_id for old_id, new_id in pairs}
            memories = await self._ensure_memories_loaded()
            now = datetime.now(UTC)
            marked: list[str] = []

            for m in memories:
                if m.id in supersede_map:
                    m.superseded_at = now
                    m.superseded_by_id = supersede_map[m.id]
                    marked.append(m.id)

            if marked:
                await self._memories_jsonl.rewrite(memories)
                self._invalidate_memories_cache()
                logger.info(
                    "batch_memories_superseded",
                    extra={"count": len(marked)},
                )

        return marked

    async def update_memory(self, entry: MemoryEntry) -> bool:
        """Update a memory entry in place."""
        async with self._write_lock:
            memories = await self._ensure_memories_loaded()

            found = False
            for i, m in enumerate(memories):
                if m.id == entry.id:
                    memories[i] = entry
                    found = True
                    break

            if found:
                await self._memories_jsonl.rewrite(memories)
                self._invalidate_memories_cache()

        return found

    async def batch_update_memories(self, entries: list[MemoryEntry]) -> int:
        """Update multiple memory entries in a single rewrite.

        Args:
            entries: List of updated MemoryEntry objects.

        Returns:
            Number of entries that were actually updated.
        """
        if not entries:
            return 0

        async with self._write_lock:
            updates = {e.id: e for e in entries}
            memories = await self._ensure_memories_loaded()
            count = 0

            for i, m in enumerate(memories):
                if m.id in updates:
                    memories[i] = updates[m.id]
                    count += 1

            if count > 0:
                await self._memories_jsonl.rewrite(memories)
                self._invalidate_memories_cache()

        return count

    async def delete_memory(
        self,
        memory_id: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> bool:
        """Archive a memory by ID (sets archived_at instead of removing)."""
        async with self._write_lock:
            memories = await self._ensure_memories_loaded()

            memory_to_delete = None
            for m in memories:
                if m.id == memory_id and self._is_active(m):
                    memory_to_delete = m
                    break

            if not memory_to_delete:
                return False

            if owner_user_id or chat_id:
                is_owner = (
                    owner_user_id and memory_to_delete.owner_user_id == owner_user_id
                )
                is_group_member = (
                    memory_to_delete.owner_user_id is None
                    and memory_to_delete.chat_id == chat_id
                )
                if not (is_owner or is_group_member):
                    return False

            memory_to_delete.archived_at = datetime.now(UTC)
            memory_to_delete.archive_reason = "user_deleted"
            await self._memories_jsonl.rewrite(memories)
            self._invalidate_memories_cache()

        logger.info("memory_deleted", extra={"memory_id": memory_id})
        return True

    # ------------------------------------------------------------------
    # Archive operations
    # ------------------------------------------------------------------

    async def archive_memories(self, memory_ids: set[str], reason: str) -> list[str]:
        """Archive memories in-place by setting archived_at.

        Args:
            memory_ids: IDs of memories to archive.
            reason: Archive reason string.

        Returns:
            List of IDs that were actually archived.
        """
        if not memory_ids:
            return []

        async with self._write_lock:
            memories = await self._ensure_memories_loaded()
            now = datetime.now(UTC)
            archived: list[str] = []

            for memory in memories:
                if memory.id in memory_ids and self._is_active(memory):
                    memory.archived_at = now
                    memory.archive_reason = reason
                    archived.append(memory.id)

            if archived:
                await self._memories_jsonl.rewrite(memories)
                self._invalidate_memories_cache()

        return archived

    async def get_archived_memories(self) -> list[MemoryEntry]:
        """Get all archived memories (from the single file)."""
        memories = await self._ensure_memories_loaded()
        return [m for m in memories if m.archived_at is not None]

    async def get_supersession_chain(self, memory_id: str) -> list[MemoryEntry]:
        """Get the chain of superseded memories leading to this ID."""
        memories = await self._ensure_memories_loaded()

        chains: dict[str, list[MemoryEntry]] = {}
        for m in memories:
            if m.superseded_by_id:
                if m.superseded_by_id not in chains:
                    chains[m.superseded_by_id] = []
                chains[m.superseded_by_id].append(m)

        result: list[MemoryEntry] = []
        current_id = memory_id

        while current_id in chains:
            predecessors = chains[current_id]
            result.extend(predecessors)
            if predecessors:
                current_id = predecessors[0].id
            else:
                break

        result.reverse()
        return result

    # ------------------------------------------------------------------
    # Garbage collection
    # ------------------------------------------------------------------

    def _should_archive(self, memory: MemoryEntry, now: datetime) -> str | None:
        """Return archive reason if memory should be archived, else None."""
        if memory.superseded_at:
            return "superseded"
        if memory.expires_at and memory.expires_at <= now:
            return "expired"
        if not memory.expires_at and memory.memory_type in EPHEMERAL_TYPES:
            if memory.created_at:
                age_days = (now - memory.created_at).days
                default_ttl = TYPE_TTL.get(memory.memory_type, 30)
                if age_days > default_ttl:
                    return "ephemeral_decay"
        return None

    async def gc(self, now: datetime | None = None) -> GCResult:
        """Garbage collect expired and superseded memories (archive in-place)."""
        if now is None:
            now = datetime.now(UTC)

        async with self._write_lock:
            memories = await self._ensure_memories_loaded()
            archived_ids: list[str] = []

            for memory in memories:
                if not self._is_active(memory):
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
                await self._memories_jsonl.rewrite(memories)
                self._invalidate_memories_cache()

        logger.info(
            "gc_complete",
            extra={"removed_count": len(archived_ids)},
        )

        return GCResult(removed_count=len(archived_ids), archived_ids=archived_ids)

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    async def compact(self, older_than_days: int = 90) -> int:
        """Permanently remove archived entries older than N days.

        Also cleans up orphaned embeddings in embeddings.jsonl for
        the removed memory IDs.

        Args:
            older_than_days: Only compact entries archived more than this many days ago.

        Returns:
            Number of entries removed.
        """
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)

        async with self._write_lock:
            memories = await self._ensure_memories_loaded()

            before_count = len(memories)
            removed_ids: set[str] = set()
            remaining: list[MemoryEntry] = []
            for m in memories:
                if m.archived_at is not None and m.archived_at < cutoff:
                    removed_ids.add(m.id)
                else:
                    remaining.append(m)

            removed = before_count - len(remaining)

            if removed > 0:
                await self._memories_jsonl.rewrite(remaining)
                self._invalidate_memories_cache()

                # Clean up orphaned embeddings
                all_embeddings = await self._embeddings_jsonl.load_all()
                clean_embeddings = [
                    e for e in all_embeddings if e.memory_id not in removed_ids
                ]
                if len(clean_embeddings) < len(all_embeddings):
                    await self._embeddings_jsonl.rewrite(clean_embeddings)

                logger.info(
                    "compact_complete",
                    extra={
                        "removed_count": removed,
                        "embeddings_cleaned": len(all_embeddings)
                        - len(clean_embeddings),
                    },
                )

        return removed

    # ------------------------------------------------------------------
    # Subject remapping
    # ------------------------------------------------------------------

    async def remap_subject_person_id(self, old_id: str, new_id: str) -> int:
        """Replace old_id with new_id in all subject_person_ids."""
        async with self._write_lock:
            memories = await self._ensure_memories_loaded()

            count = 0
            for memory in memories:
                if old_id in memory.subject_person_ids:
                    memory.subject_person_ids = [
                        new_id if pid == old_id else pid
                        for pid in memory.subject_person_ids
                    ]
                    count += 1

            if count > 0:
                await self._memories_jsonl.rewrite(memories)
                self._invalidate_memories_cache()
                logger.debug(
                    "remapped_subject_person_id",
                    extra={"old_id": old_id, "new_id": new_id, "count": count},
                )

        return count

    async def get_all_memories(self) -> list[MemoryEntry]:
        """Get all memories including archived, expired, and superseded."""
        return await self._ensure_memories_loaded()

    async def clear(self) -> int:
        """Physically remove all memories and embeddings.

        Unlike delete_memory (which archives), this permanently wipes
        both memories.jsonl and embeddings.jsonl.

        Returns:
            Number of entries removed.
        """
        async with self._write_lock:
            memories = await self._ensure_memories_loaded()
            count = len(memories)

            if count > 0:
                await self._memories_jsonl.rewrite([])
                self._invalidate_memories_cache()

            # Also clear embeddings
            all_embeddings = await self._embeddings_jsonl.load_all()
            if all_embeddings:
                await self._embeddings_jsonl.rewrite([])

        logger.info("clear_complete", extra={"removed_count": count})
        return count

    # ------------------------------------------------------------------
    # Embedding storage
    # ------------------------------------------------------------------

    async def save_embedding(self, memory_id: str, embedding_base64: str) -> None:
        """Append an embedding record to embeddings.jsonl."""
        record = EmbeddingRecord(memory_id=memory_id, embedding=embedding_base64)
        await self._embeddings_jsonl.append(record)

    async def load_embeddings(self) -> dict[str, str]:
        """Load all embeddings as {memory_id: base64} mapping."""
        records = await self._embeddings_jsonl.load_all()
        # Last-write wins for duplicates
        return {r.memory_id: r.embedding for r in records}

    async def get_embedding_for_memory(self, memory_id: str) -> str | None:
        """Get embedding for a single memory."""
        embeddings = await self.load_embeddings()
        return embeddings.get(memory_id)
