"""Memory CRUD operations: create, read, update, delete."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from ash.memory.secrets import contains_secret
from ash.store.mappers import row_to_memory as _row_to_memory
from ash.store.memories.helpers import load_subjects_batch, row_to_memory_full
from ash.store.types import (
    MemoryEntry,
    MemoryType,
    Sensitivity,
)

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)


class MemoryCrudMixin:
    """Memory create, read, update, delete operations."""

    async def _resolve_subject_name(self: Store, person_ids: list[str]) -> str | None:
        if not person_ids:
            return None

        # Check cache first, collect misses for lookup
        names: list[str] = []
        cache_misses: list[str] = []

        for pid in person_ids:
            if pid in self._person_name_cache:
                names.append(self._person_name_cache[pid])
            else:
                cache_misses.append(pid)

        # Fetch any cache misses - try batch first, fall back to individual
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
                    # Fall back to get_person for individual lookup
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
            return await row_to_memory_full(session, row)

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
                return await row_to_memory_full(session, row)

            # Try prefix match
            result = await session.execute(
                text(
                    "SELECT * FROM memories WHERE id LIKE :prefix AND archived_at IS NULL"
                ),
                {"prefix": f"{prefix}%"},
            )
            rows = result.fetchall()
            if len(rows) == 1:
                return await row_to_memory_full(session, rows[0])
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

            # Scope filtering in SQL so LIMIT returns the correct count
            if owner_user_id and chat_id:
                conditions.append(
                    "(owner_user_id = :owner_user_id OR (owner_user_id IS NULL AND chat_id = :chat_id))"
                )
                params["owner_user_id"] = owner_user_id
                params["chat_id"] = chat_id
            elif owner_user_id:
                conditions.append("owner_user_id = :owner_user_id")
                params["owner_user_id"] = owner_user_id
            elif chat_id:
                conditions.append("(owner_user_id IS NULL AND chat_id = :chat_id)")
                params["chat_id"] = chat_id

            where = " AND ".join(conditions)
            query = f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC"
            if limit is not None:
                query += " LIMIT :limit"
                params["limit"] = limit

            result = await session.execute(text(query), params)
            rows = result.fetchall()

            # Load subjects in batch
            memory_ids = [row.id for row in rows]
            subjects_map = await load_subjects_batch(session, memory_ids)

            memories = []
            for row in rows:
                memory = _row_to_memory(row)
                memory.subject_person_ids = subjects_map.get(memory.id, [])
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

    async def get_all_memories(self: Store) -> list[MemoryEntry]:
        """Get all memories including archived."""
        async with self._db.session() as session:
            result = await session.execute(text("SELECT * FROM memories"))
            rows = result.fetchall()
            memory_ids = [row.id for row in rows]
            subjects_map = await load_subjects_batch(session, memory_ids)

            memories = []
            for row in rows:
                memory = _row_to_memory(row)
                memory.subject_person_ids = subjects_map.get(memory.id, [])
                memories.append(memory)
            return memories
