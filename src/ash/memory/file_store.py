"""Filesystem-based memory store.

Primary storage for memories and people, with in-memory caching
and mtime-based invalidation.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ash.config.paths import (
    get_memories_jsonl_path,
    get_memory_archive_path,
    get_people_jsonl_path,
)
from ash.memory.jsonl import ArchiveJSONL, MemoryJSONL, PersonJSONL
from ash.memory.types import (
    EPHEMERAL_TYPES,
    TYPE_TTL,
    GCResult,
    MemoryEntry,
    MemoryType,
    PersonEntry,
    matches_scope,
)

logger = logging.getLogger(__name__)


class FileMemoryStore:
    """Filesystem-based store for memories and people.

    Uses JSONL files as the source of truth with in-memory caching
    for fast reads. Cache is invalidated based on file mtime.
    """

    def __init__(
        self,
        memories_path: Path | None = None,
        archive_path: Path | None = None,
        people_path: Path | None = None,
    ) -> None:
        """Initialize the file store.

        Args:
            memories_path: Path to memories.jsonl (default: ~/.ash/memory/memories.jsonl)
            archive_path: Path to archive.jsonl (default: ~/.ash/memory/archive.jsonl)
            people_path: Path to people.jsonl (default: ~/.ash/memory/people.jsonl)
        """
        self._memories_jsonl = MemoryJSONL(memories_path or get_memories_jsonl_path())
        self._archive_jsonl = ArchiveJSONL(archive_path or get_memory_archive_path())
        self._people_jsonl = PersonJSONL(people_path or get_people_jsonl_path())

        # In-memory cache
        self._memories_cache: list[MemoryEntry] | None = None
        self._memories_mtime: float | None = None
        self._people_cache: list[PersonEntry] | None = None
        self._people_mtime: float | None = None

    async def _ensure_memories_loaded(self) -> list[MemoryEntry]:
        """Load memories from disk if cache is stale."""
        current_mtime = self._memories_jsonl.get_mtime()

        if self._memories_cache is None or current_mtime != self._memories_mtime:
            self._memories_cache = await self._memories_jsonl.load_all()
            self._memories_mtime = current_mtime

        return self._memories_cache

    async def _ensure_people_loaded(self) -> list[PersonEntry]:
        """Load people from disk if cache is stale."""
        current_mtime = (
            self._people_jsonl.get_mtime() if self._people_jsonl.exists() else None
        )

        if self._people_cache is None or current_mtime != self._people_mtime:
            self._people_cache = await self._people_jsonl.load_all()
            self._people_mtime = current_mtime

        return self._people_cache

    def _invalidate_memories_cache(self) -> None:
        """Invalidate memories cache after write."""
        self._memories_cache = None
        self._memories_mtime = None

    def _invalidate_people_cache(self) -> None:
        """Invalidate people cache after write."""
        self._people_cache = None
        self._people_mtime = None

    # Memory CRUD operations

    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.KNOWLEDGE,
        embedding: str = "",
        source: str = "user",
        expires_at: datetime | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        subject_person_ids: list[str] | None = None,
        observed_at: datetime | None = None,
        source_user_id: str | None = None,
        source_user_name: str | None = None,
        source_session_id: str | None = None,
        source_message_id: str | None = None,
        extraction_confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Add a new memory entry.

        Args:
            content: The memory content.
            memory_type: Type classification for lifecycle.
            embedding: Base64-encoded embedding.
            source: Origin tracking.
            expires_at: Explicit expiration time.
            owner_user_id: Personal scope owner.
            chat_id: Group scope.
            subject_person_ids: People this memory is about.
            observed_at: When fact was observed (for extraction).
            source_user_id: Who said/provided this fact (for multi-user attribution).
            source_user_name: Display name of source user.
            source_session_id: Session ID for extraction tracing.
            source_message_id: Message ID for extraction tracing.
            extraction_confidence: Confidence score for extraction.
            metadata: Additional metadata.

        Returns:
            The created memory entry.
        """
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            version=1,
            content=content,
            memory_type=memory_type,
            embedding=embedding,
            created_at=datetime.now(UTC),
            observed_at=observed_at,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            subject_person_ids=subject_person_ids or [],
            source=source,
            source_user_id=source_user_id,
            source_user_name=source_user_name,
            source_session_id=source_session_id,
            source_message_id=source_message_id,
            extraction_confidence=extraction_confidence,
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
        """Get memory by ID.

        Args:
            memory_id: Memory UUID.

        Returns:
            Memory entry or None if not found.
        """
        memories = await self._ensure_memories_loaded()
        for m in memories:
            if m.id == memory_id:
                return m
        return None

    async def get_memory_by_prefix(self, memory_id_prefix: str) -> MemoryEntry | None:
        """Get memory by ID or ID prefix.

        Supports short IDs like git - if exactly one memory matches the prefix,
        return it. Returns None if no match or multiple matches.

        Args:
            memory_id_prefix: Full ID or prefix.

        Returns:
            Memory entry or None if not found or ambiguous.
        """
        memories = await self._ensure_memories_loaded()

        # Try exact match first
        for m in memories:
            if m.id == memory_id_prefix:
                return m

        # Try prefix match
        matches = [m for m in memories if m.id.startswith(memory_id_prefix)]
        if len(matches) == 1:
            return matches[0]
        return None

    async def get_memories(
        self,
        limit: int = 100,
        include_expired: bool = False,
        include_superseded: bool = False,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[MemoryEntry]:
        """Get memory entries with filters.

        Args:
            limit: Maximum entries to return.
            include_expired: Include expired entries.
            include_superseded: Include superseded entries.
            owner_user_id: Filter to user's personal memories.
            chat_id: Filter to group memories for this chat.

        Returns:
            List of memory entries, newest first.
        """
        memories = await self._ensure_memories_loaded()
        now = datetime.now(UTC)

        result: list[MemoryEntry] = []
        for m in memories:
            # Expiration filter
            if not include_expired and m.expires_at and m.expires_at <= now:
                continue

            # Supersession filter
            if not include_superseded and m.superseded_at:
                continue

            # Scope filter
            if not matches_scope(m, owner_user_id, chat_id):
                continue

            result.append(m)

        # Sort by created_at descending
        result.sort(
            key=lambda x: x.created_at or datetime.min.replace(tzinfo=UTC), reverse=True
        )

        return result[:limit]

    async def get_memories_about_person(
        self,
        person_id: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        limit: int = 50,
        include_expired: bool = False,
        include_superseded: bool = False,
    ) -> list[MemoryEntry]:
        """Get memories about a specific person.

        Args:
            person_id: Person UUID.
            owner_user_id: Filter to user's personal memories.
            chat_id: Filter to group memories.
            limit: Maximum entries.
            include_expired: Include expired.
            include_superseded: Include superseded.

        Returns:
            List of memories about this person.
        """
        memories = await self.get_memories(
            limit=1000,  # Get more to filter
            include_expired=include_expired,
            include_superseded=include_superseded,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
        )

        result = [m for m in memories if person_id in m.subject_person_ids]
        return result[:limit]

    async def mark_memory_superseded(
        self,
        memory_id: str,
        superseded_by_id: str,
    ) -> bool:
        """Mark a memory as superseded by another memory.

        Args:
            memory_id: Memory to mark as superseded.
            superseded_by_id: Memory that supersedes it.

        Returns:
            True if memory was found and marked.
        """
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

    async def update_memory(self, entry: MemoryEntry) -> bool:
        """Update a memory entry in place.

        Args:
            entry: Updated memory entry (matched by ID).

        Returns:
            True if memory was found and updated.
        """
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

    async def delete_memory(
        self,
        memory_id: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: Memory UUID.
            owner_user_id: Required owner for authorization check.
            chat_id: Required chat for authorization check.

        Returns:
            True if memory was found and deleted.
        """
        memories = await self._ensure_memories_loaded()

        memory_to_delete = None
        for m in memories:
            if m.id == memory_id:
                memory_to_delete = m
                break

        if not memory_to_delete:
            return False

        # Authorization check
        if owner_user_id or chat_id:
            is_owner = owner_user_id and memory_to_delete.owner_user_id == owner_user_id
            is_group_member = (
                memory_to_delete.owner_user_id is None
                and memory_to_delete.chat_id == chat_id
            )
            if not (is_owner or is_group_member):
                return False

        # Remove from memories
        new_memories = [m for m in memories if m.id != memory_id]
        await self._memories_jsonl.rewrite(new_memories)
        self._invalidate_memories_cache()

        logger.info("memory_deleted", extra={"memory_id": memory_id})
        return True

    async def gc(self, now: datetime | None = None) -> GCResult:
        """Garbage collect expired and superseded memories.

        Moves removed memories to archive with reason.

        Args:
            now: Current time (defaults to UTC now).

        Returns:
            GC result with counts.
        """
        if now is None:
            now = datetime.now(UTC)

        memories = await self._ensure_memories_loaded()
        to_remove: list[tuple[MemoryEntry, str]] = []

        for memory in memories:
            # Explicit supersession
            if memory.superseded_at:
                to_remove.append((memory, "superseded"))
                continue

            # Explicit expiration
            if memory.expires_at and memory.expires_at <= now:
                to_remove.append((memory, "expired"))
                continue

            # Ephemeral memory decay (no explicit expiration)
            if not memory.expires_at and memory.memory_type in EPHEMERAL_TYPES:
                if memory.created_at:
                    age_days = (now - memory.created_at).days
                    default_ttl = TYPE_TTL.get(memory.memory_type, 30)

                    if age_days > default_ttl:
                        to_remove.append((memory, "ephemeral_decay"))
                        continue

        if not to_remove:
            return GCResult(removed_count=0)

        # Archive before removing
        archived_ids: list[str] = []
        for memory, reason in to_remove:
            memory.archived_at = now
            memory.archive_reason = reason
            await self._archive_jsonl.append(memory)
            archived_ids.append(memory.id)
            logger.info(
                "gc_archive_memory",
                extra={"memory_id": memory.id, "reason": reason},
            )

        # Compact: rewrite file without removed memories
        remove_ids = {m.id for m, _ in to_remove}
        remaining = [m for m in memories if m.id not in remove_ids]
        await self._memories_jsonl.rewrite(remaining)
        self._invalidate_memories_cache()

        logger.info(
            "gc_complete",
            extra={"removed_count": len(to_remove)},
        )

        return GCResult(removed_count=len(to_remove), archived_ids=archived_ids)

    async def get_all_memories(self) -> list[MemoryEntry]:
        """Get all memories including expired and superseded.

        Used for migration and index rebuild.

        Returns:
            All memory entries.
        """
        return await self._ensure_memories_loaded()

    # Person CRUD operations

    async def create_person(
        self,
        owner_user_id: str,
        name: str,
        relationship: str | None = None,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PersonEntry:
        """Create a new person entity.

        Args:
            owner_user_id: Who owns this person record.
            name: Primary name.
            relationship: Relationship to owner.
            aliases: Alternate names.
            metadata: Additional metadata.

        Returns:
            The created person entry.
        """
        now = datetime.now(UTC)
        entry = PersonEntry(
            id=str(uuid.uuid4()),
            version=1,
            owner_user_id=owner_user_id,
            name=name,
            relationship=relationship,
            aliases=aliases or [],
            created_at=now,
            updated_at=now,
            metadata=metadata,
        )

        await self._people_jsonl.append(entry)
        self._invalidate_people_cache()

        logger.debug("person_created", extra={"person_id": entry.id, "name": name})
        return entry

    async def get_person(
        self,
        person_id: str,
        owner_user_id: str | None = None,
    ) -> PersonEntry | None:
        """Get person by ID.

        Args:
            person_id: Person UUID.
            owner_user_id: Optional owner filter.

        Returns:
            Person entry or None.
        """
        people = await self._ensure_people_loaded()

        for p in people:
            if p.id == person_id:
                if owner_user_id and p.owner_user_id != owner_user_id:
                    return None
                return p

        return None

    def _normalize_reference(self, text: str) -> str:
        """Normalize a person reference by removing common prefixes."""
        result = text.lower().strip()
        for prefix in ["my ", "the ", "@"]:
            if result.startswith(prefix):
                return result[len(prefix) :]
        return result

    async def find_person_by_reference(
        self,
        owner_user_id: str,
        reference: str,
    ) -> PersonEntry | None:
        """Find person by name, relationship, or alias.

        Args:
            owner_user_id: Owner to search within.
            reference: Name, relationship, or alias to search.

        Returns:
            Person entry or None.
        """
        ref = self._normalize_reference(reference)
        people = await self._ensure_people_loaded()

        for person in people:
            if person.owner_user_id != owner_user_id:
                continue

            if person.name.lower() == ref:
                return person
            if person.relationship and person.relationship.lower() == ref:
                return person
            if person.aliases:
                for alias in person.aliases:
                    if self._normalize_reference(alias) == ref:
                        return person

        return None

    async def get_people_for_user(self, owner_user_id: str) -> list[PersonEntry]:
        """Get all people for a user.

        Args:
            owner_user_id: User to get people for.

        Returns:
            List of person entries, sorted by name.
        """
        people = await self._ensure_people_loaded()
        result = [p for p in people if p.owner_user_id == owner_user_id]
        result.sort(key=lambda x: x.name)
        return result

    async def update_person(
        self,
        person_id: str,
        owner_user_id: str,
        name: str | None = None,
        relationship: str | None = None,
        aliases: list[str] | None = None,
    ) -> PersonEntry | None:
        """Update person details.

        Args:
            person_id: Person UUID.
            owner_user_id: Owner for authorization.
            name: New name (if provided).
            relationship: New relationship (if provided).
            aliases: New aliases (if provided).

        Returns:
            Updated person entry or None if not found.
        """
        people = await self._ensure_people_loaded()

        person = None
        for p in people:
            if p.id == person_id and p.owner_user_id == owner_user_id:
                person = p
                break

        if not person:
            return None

        if name is not None:
            person.name = name
        if relationship is not None:
            person.relationship = relationship
        if aliases is not None:
            person.aliases = aliases
        person.updated_at = datetime.now(UTC)

        await self._people_jsonl.rewrite(people)
        self._invalidate_people_cache()

        return person

    async def add_person_alias(
        self,
        person_id: str,
        alias: str,
        owner_user_id: str,
    ) -> PersonEntry | None:
        """Add an alias to a person.

        Args:
            person_id: Person UUID.
            alias: Alias to add.
            owner_user_id: Owner for authorization.

        Returns:
            Updated person entry or None if not found.
        """
        people = await self._ensure_people_loaded()

        person = None
        for p in people:
            if p.id == person_id and p.owner_user_id == owner_user_id:
                person = p
                break

        if not person:
            return None

        aliases = list(person.aliases or [])
        if alias.lower() not in [a.lower() for a in aliases]:
            aliases.append(alias)
            person.aliases = aliases
            person.updated_at = datetime.now(UTC)

            await self._people_jsonl.rewrite(people)
            self._invalidate_people_cache()

        return person

    async def get_all_people(self) -> list[PersonEntry]:
        """Get all people.

        Returns:
            All person entries.
        """
        return await self._ensure_people_loaded()

    # Archive operations

    async def get_archived_memories(self) -> list[MemoryEntry]:
        """Get all archived memories.

        Returns:
            All archived memory entries.
        """
        return await self._archive_jsonl.load_all()

    async def get_supersession_chain(self, memory_id: str) -> list[MemoryEntry]:
        """Get the chain of superseded memories leading to this ID.

        Args:
            memory_id: Memory to trace back from.

        Returns:
            List of memories in supersession order (oldest first).
        """
        archived = await self._archive_jsonl.load_all()

        # Build reverse lookup: superseded_by_id -> memory
        chains: dict[str, list[MemoryEntry]] = {}
        for m in archived:
            if m.superseded_by_id:
                if m.superseded_by_id not in chains:
                    chains[m.superseded_by_id] = []
                chains[m.superseded_by_id].append(m)

        # Walk back from current memory
        result: list[MemoryEntry] = []
        current_id = memory_id

        while current_id in chains:
            predecessors = chains[current_id]
            # There should only be one, but handle multiple
            result.extend(predecessors)
            if predecessors:
                # Continue from the oldest predecessor
                current_id = predecessors[0].id
            else:
                break

        # Reverse to get oldest first
        result.reverse()
        return result
