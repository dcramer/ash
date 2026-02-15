"""Memory CRUD operations mixin for Store (SQLite-backed)."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from ash.memory.secrets import contains_secret
from ash.store.mappers import row_to_memory as _row_to_memory
from ash.store.types import (
    EPHEMERAL_TYPES,
    TYPE_TTL,
    GCResult,
    MemoryEntry,
    MemoryType,
    Sensitivity,
    matches_scope,
)

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)


async def _load_subjects(session, memory_id: str) -> list[str]:
    """Load subject_person_ids for a memory."""
    result = await session.execute(
        text("SELECT person_id FROM memory_subjects WHERE memory_id = :id"),
        {"id": memory_id},
    )
    return [row[0] for row in result.fetchall()]


async def _load_subjects_batch(session, memory_ids: list[str]) -> dict[str, list[str]]:
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


async def _row_to_memory_full(session, row) -> MemoryEntry:
    """Convert a row to a MemoryEntry with subject_person_ids loaded."""
    memory = _row_to_memory(row)
    memory.subject_person_ids = await _load_subjects(session, memory.id)
    return memory


class MemoryOpsMixin:
    """Memory CRUD, eviction, and lifecycle operations."""

    async def _resolve_subject_name(self: Store, person_ids: list[str]) -> str | None:
        if not person_ids:
            return None
        names: list[str] = []
        for pid in person_ids:
            if pid in self._person_name_cache:
                names.append(self._person_name_cache[pid])
                continue
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

        async with self._db.session() as session:
            await session.execute(
                text("""
                    INSERT INTO memories (id, version, content, memory_type, source,
                        owner_user_id, chat_id, source_username, source_display_name,
                        source_session_id, source_message_id, extraction_confidence,
                        sensitivity, portable, created_at, observed_at, expires_at,
                        metadata)
                    VALUES (:id, 1, :content, :memory_type, :source,
                        :owner_user_id, :chat_id, :source_username, :source_display_name,
                        :source_session_id, :source_message_id, :extraction_confidence,
                        :sensitivity, :portable, :created_at, :observed_at, :expires_at,
                        :metadata)
                """),
                {
                    "id": memory_id,
                    "content": content,
                    "memory_type": memory_type.value,
                    "source": source,
                    "owner_user_id": owner_user_id,
                    "chat_id": chat_id,
                    "source_username": source_username,
                    "source_display_name": source_display_name,
                    "source_session_id": source_session_id,
                    "source_message_id": source_message_id,
                    "extraction_confidence": extraction_confidence,
                    "sensitivity": sensitivity.value if sensitivity else None,
                    "portable": 1 if portable else 0,
                    "created_at": now.isoformat(),
                    "observed_at": observed_at.isoformat() if observed_at else None,
                    "expires_at": expires_at.isoformat() if expires_at else None,
                    "metadata": json.dumps(metadata) if metadata else None,
                },
            )
            for pid in subject_pids:
                await session.execute(
                    text(
                        "INSERT INTO memory_subjects (memory_id, person_id) VALUES (:mid, :pid)"
                    ),
                    {"mid": memory_id, "pid": pid},
                )

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

        if embedding_floats:
            embedding_base64 = MemoryEntry.encode_embedding(embedding_floats)
            try:
                await self._index.add_embedding(memory.id, embedding_floats)
            except Exception:
                logger.warning("Failed to index memory, continuing", exc_info=True)
            # Store base64 embedding on the entry for callers that need it
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
        async with self._db.session() as session:
            result = await session.execute(
                text("SELECT * FROM memories WHERE id = :id AND archived_at IS NULL"),
                {"id": memory_id},
            )
            row = result.fetchone()
            if not row:
                return None
            return await _row_to_memory_full(session, row)

    async def get_memory_by_prefix(self: Store, prefix: str) -> MemoryEntry | None:
        """Find a memory by ID prefix match."""
        async with self._db.session() as session:
            # Try exact match first
            result = await session.execute(
                text("SELECT * FROM memories WHERE id = :id AND archived_at IS NULL"),
                {"id": prefix},
            )
            row = result.fetchone()
            if row:
                return await _row_to_memory_full(session, row)

            # Try prefix match
            result = await session.execute(
                text(
                    "SELECT * FROM memories WHERE id LIKE :prefix AND archived_at IS NULL"
                ),
                {"prefix": f"{prefix}%"},
            )
            rows = result.fetchall()
            if len(rows) == 1:
                return await _row_to_memory_full(session, rows[0])
            return None

    async def list_memories(
        self: Store,
        limit: int | None = 20,
        include_expired: bool = False,
        include_superseded: bool = False,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[MemoryEntry]:
        async with self._db.session() as session:
            conditions = ["archived_at IS NULL"]
            params: dict[str, Any] = {}

            if not include_expired:
                conditions.append("(expires_at IS NULL OR expires_at > :now)")
                params["now"] = datetime.now(UTC).isoformat()

            if not include_superseded:
                conditions.append("superseded_at IS NULL")

            where = " AND ".join(conditions)
            query = f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC"
            if limit is not None:
                query += " LIMIT :limit"
                params["limit"] = limit

            result = await session.execute(text(query), params)
            rows = result.fetchall()

            # Load subjects in batch
            memory_ids = [row.id for row in rows]
            subjects_map = await _load_subjects_batch(session, memory_ids)

            memories = []
            for row in rows:
                memory = _row_to_memory(row)
                memory.subject_person_ids = subjects_map.get(memory.id, [])
                # Apply scope filter in Python (complex logic)
                if not matches_scope(memory, owner_user_id, chat_id):
                    continue
                memories.append(memory)

            return memories

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

        now = datetime.now(UTC).isoformat()
        async with self._db.session() as session:
            await session.execute(
                text(
                    "UPDATE memories SET archived_at = :now, archive_reason = 'user_deleted' WHERE id = :id"
                ),
                {"now": now, "id": full_id},
            )

        try:
            await self._index.delete_embedding(full_id)
        except Exception:
            logger.warning(
                "Failed to delete memory embedding",
                extra={"memory_id": full_id},
                exc_info=True,
            )

        logger.info("memory_deleted", extra={"memory_id": full_id})
        return True

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
            if memory.created_at:
                age_days = (now - memory.created_at).days
                default_ttl = TYPE_TTL.get(memory.memory_type, 30)
                if age_days > default_ttl:
                    return "ephemeral_decay"
        return None

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

    async def rebuild_index(self: Store) -> int:
        """Rebuild vector index, generating embeddings for any memories missing them."""
        async with self._db.session() as session:
            result = await session.execute(
                text(
                    "SELECT id, content FROM memories WHERE archived_at IS NULL AND superseded_at IS NULL"
                ),
            )
            active_memories = result.fetchall()

        # Check which have embeddings
        indexed_count = await self._index.get_embedding_count()

        generated = 0
        for row in active_memories:
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
                extra={"generated": generated, "indexed_before": indexed_count},
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
                    memory.subject_person_ids = await _load_subjects(session, memory.id)
                    result.append(memory)
                current_id = predecessors[0].id

            result.reverse()
            return result

    async def batch_update_memories(self: Store, entries: list[MemoryEntry]) -> int:
        """Update multiple memories."""
        if not entries:
            return 0

        count = 0
        async with self._db.session() as session:
            for entry in entries:
                r = await session.execute(
                    text("""
                        UPDATE memories SET
                            content = :content,
                            memory_type = :memory_type,
                            source = :source,
                            owner_user_id = :owner_user_id,
                            chat_id = :chat_id,
                            source_username = :source_username,
                            source_display_name = :source_display_name,
                            sensitivity = :sensitivity,
                            portable = :portable,
                            expires_at = :expires_at,
                            superseded_at = :superseded_at,
                            superseded_by_id = :superseded_by_id,
                            archived_at = :archived_at,
                            archive_reason = :archive_reason,
                            metadata = :metadata
                        WHERE id = :id
                    """),
                    {
                        "id": entry.id,
                        "content": entry.content,
                        "memory_type": entry.memory_type.value,
                        "source": entry.source,
                        "owner_user_id": entry.owner_user_id,
                        "chat_id": entry.chat_id,
                        "source_username": entry.source_username,
                        "source_display_name": entry.source_display_name,
                        "sensitivity": entry.sensitivity.value
                        if entry.sensitivity
                        else None,
                        "portable": 1 if entry.portable else 0,
                        "expires_at": entry.expires_at.isoformat()
                        if entry.expires_at
                        else None,
                        "superseded_at": entry.superseded_at.isoformat()
                        if entry.superseded_at
                        else None,
                        "superseded_by_id": entry.superseded_by_id,
                        "archived_at": entry.archived_at.isoformat()
                        if entry.archived_at
                        else None,
                        "archive_reason": entry.archive_reason,
                        "metadata": json.dumps(entry.metadata)
                        if entry.metadata
                        else None,
                    },
                )
                if r.rowcount > 0:
                    count += 1

                    # Update subjects
                    await session.execute(
                        text("DELETE FROM memory_subjects WHERE memory_id = :id"),
                        {"id": entry.id},
                    )
                    for pid in entry.subject_person_ids:
                        await session.execute(
                            text(
                                "INSERT INTO memory_subjects (memory_id, person_id) VALUES (:mid, :pid)"
                            ),
                            {"mid": entry.id, "pid": pid},
                        )

        return count

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
                if r.rowcount > 0:
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

    async def get_all_memories(self: Store) -> list[MemoryEntry]:
        """Get all memories including archived."""
        async with self._db.session() as session:
            result = await session.execute(text("SELECT * FROM memories"))
            rows = result.fetchall()
            memory_ids = [row.id for row in rows]
            subjects_map = await _load_subjects_batch(session, memory_ids)

            memories = []
            for row in rows:
                memory = _row_to_memory(row)
                memory.subject_person_ids = subjects_map.get(memory.id, [])
                memories.append(memory)
            return memories
