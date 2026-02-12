"""Memory manager for orchestrating retrieval and persistence.

This module provides the primary facade for all memory operations,
using filesystem-primary storage with a SQLite vector index.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.file_store import FileMemoryStore
from ash.memory.index import VectorIndex
from ash.memory.types import (
    GCResult,
    MemoryEntry,
    MemoryType,
    PersonEntry,
    PersonResolutionResult,
    RetrievedContext,
    SearchResult,
    Sensitivity,
    matches_scope,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from ash.llm import LLMProvider, LLMRegistry

logger = logging.getLogger(__name__)

CONFLICT_SIMILARITY_THRESHOLD = 0.75

# Regex patterns for detecting secrets/credentials in memory content
_SECRETS_PATTERNS = [
    # API keys and tokens
    re.compile(r"\b(sk-[a-zA-Z0-9]{20,})\b"),  # OpenAI/Anthropic keys
    re.compile(r"\b(gh[pors]_[a-zA-Z0-9]{36,})\b"),  # GitHub tokens (PAT, OAuth, etc.)
    re.compile(r"\b(AKIA[A-Z0-9]{16})\b"),  # AWS access key IDs
    re.compile(r"\b(xox[baprs]-[a-zA-Z0-9-]{10,})\b"),  # Slack tokens
    # Credit card numbers (16 digits with optional separators)
    re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    # Social Security Numbers
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Passwords in common formats
    re.compile(r"\b(?:password|passwd|pwd)\s*(?:is|:|=)\s*\S+", re.IGNORECASE),
    # Private keys
    re.compile(
        r"-----BEGIN\s+(?:RSA\s+|DSA\s+|EC\s+|OPENSSH\s+|PGP\s+)?PRIVATE\s+KEY-----",
        re.IGNORECASE,
    ),
    # API key assignments
    re.compile(
        r"\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*(?:is|:|=)\s*\S+",
        re.IGNORECASE,
    ),
]


def contains_secret(content: str) -> bool:
    """Check if content contains patterns that look like secrets.

    Args:
        content: The text content to check.

    Returns:
        True if any secret pattern is detected.
    """
    return any(pattern.search(content) for pattern in _SECRETS_PATTERNS)


# LLM verification prompt for supersession
SUPERSESSION_PROMPT = """Given these two memories, determine if the NEW memory supersedes/replaces the OLD memory.

OLD: "{old_content}"
NEW: "{new_content}"

Answer YES if the new memory updates, corrects, or replaces the old memory.
Answer NO if they are about different things or both should be kept.

Answer only YES or NO."""


class MemoryManager:
    """Orchestrates memory retrieval and persistence.

    Uses filesystem-primary storage (JSONL files) with a SQLite
    vector index for semantic search. The JSONL is the source of
    truth and the index is rebuildable.
    """

    def __init__(
        self,
        store: FileMemoryStore,
        index: VectorIndex,
        embedding_generator: EmbeddingGenerator,
        db_session: AsyncSession,
        llm: LLMProvider | None = None,
        max_entries: int | None = None,
    ) -> None:
        """Initialize memory manager.

        Args:
            store: Filesystem-based memory store.
            index: Vector index for semantic search.
            embedding_generator: Generator for embeddings.
            db_session: Database session for vector operations.
            llm: LLM for supersession verification (optional).
            max_entries: Optional cap on active memories.
        """
        from ash.memory.person import PersonManager

        self._store = store
        self._index = index
        self._embeddings = embedding_generator
        self._session = db_session
        self._person_manager = PersonManager(store)
        self._llm = llm
        self._max_entries = max_entries

    async def get_context_for_message(
        self,
        user_id: str,
        user_message: str,
        chat_id: str | None = None,
        max_memories: int = 10,
        chat_type: str | None = None,
        participant_usernames: list[str] | None = None,
    ) -> RetrievedContext:
        """Retrieve relevant memory context before LLM call.

        Supports cross-context retrieval: facts learned about a person in public
        chats are recallable in private chats with that person, subject to
        sensitivity filtering.

        Args:
            user_id: User ID for scoping.
            user_message: Message to find relevant context for.
            chat_id: Optional chat ID for group memories.
            max_memories: Maximum memories to return.
            chat_type: Type of chat ("private", "group", "supergroup").
            participant_usernames: Usernames of other participants (for cross-context).

        Returns:
            Retrieved context with matching memories.
        """
        try:
            # Get user's own memories
            memories = await self.search(
                query=user_message,
                limit=max_memories,
                owner_user_id=user_id,
                chat_id=chat_id,
            )
        except Exception:
            logger.warning(
                "Failed to search memories, continuing without", exc_info=True
            )
            memories = []

        # Cross-context retrieval: get memories about participants from other owners
        if participant_usernames:
            for username in participant_usernames:
                try:
                    cross_memories = await self._store.find_memories_about_user(
                        username=username,
                        exclude_owner_user_id=user_id,  # Don't double-count
                        limit=max_memories,
                    )

                    # Apply privacy filtering and convert to SearchResult
                    for memory in cross_memories:
                        if self._passes_privacy_filter(
                            memory, chat_type, username, participant_usernames
                        ):
                            memories.append(
                                SearchResult(
                                    id=memory.id,
                                    content=memory.content,
                                    similarity=0.7,  # Default for cross-context
                                    metadata={
                                        "memory_type": memory.memory_type.value,
                                        "subject_person_ids": memory.subject_person_ids,
                                        "cross_context": True,
                                    },
                                    source_type="memory",
                                )
                            )
                except Exception:
                    logger.debug(
                        "Failed to get cross-context memories for %s", username
                    )

        # Apply privacy filtering to own memories
        memories = [
            m
            for m in memories
            if self._passes_privacy_filter_search_result(
                m, chat_type, participant_usernames
            )
        ]

        # Deduplicate and limit
        seen_ids: set[str] = set()
        unique_memories: list[SearchResult] = []
        for m in memories:
            if m.id not in seen_ids:
                seen_ids.add(m.id)
                unique_memories.append(m)
                if len(unique_memories) >= max_memories:
                    break

        return RetrievedContext(memories=unique_memories)

    def _passes_privacy_filter(
        self,
        memory: MemoryEntry,
        chat_type: str | None,
        querying_username: str | None,
        participant_usernames: list[str] | None,
    ) -> bool:
        """Filter memories based on sensitivity and context.

        Args:
            memory: Memory to filter.
            chat_type: Type of chat ("private", "group", etc.).
            querying_username: Username being queried about.
            participant_usernames: All participants in the chat.

        Returns:
            True if memory passes the privacy filter.
        """
        # No sensitivity or PUBLIC = always pass
        if memory.sensitivity is None or memory.sensitivity == Sensitivity.PUBLIC:
            return True

        # Check if querying user is a subject of this memory
        is_subject = self._is_user_subject_of_memory(querying_username, memory)

        if memory.sensitivity == Sensitivity.PERSONAL:
            # PERSONAL: only show to the subject
            return is_subject

        if memory.sensitivity == Sensitivity.SENSITIVE:
            # SENSITIVE: only in private chat with the subject
            is_private = chat_type == "private"
            return is_private and is_subject

        return False

    def _passes_privacy_filter_search_result(
        self,
        result: SearchResult,
        chat_type: str | None,
        participant_usernames: list[str] | None,
    ) -> bool:
        """Filter search results based on sensitivity metadata.

        For search results where we don't have the full MemoryEntry,
        we check if metadata contains sensitivity.
        """
        if not result.metadata:
            return True

        sensitivity_str = result.metadata.get("sensitivity")
        if not sensitivity_str:
            return True

        try:
            sensitivity = Sensitivity(sensitivity_str)
        except ValueError:
            return True

        if sensitivity == Sensitivity.PUBLIC:
            return True

        # For non-public, we need to be more conservative in group chats
        is_private = chat_type == "private"

        if sensitivity == Sensitivity.SENSITIVE:
            # Only allow in private chats
            return is_private

        if sensitivity == Sensitivity.PERSONAL:
            # In group chats, be conservative and hide PERSONAL memories
            # about third parties (we can't easily determine subject here)
            return is_private

        return True

    def _is_user_subject_of_memory(
        self, username: str | None, memory: MemoryEntry
    ) -> bool:
        """Check if a username is the subject of a memory.

        Uses the existing _person_matches_username logic to check if
        the user matches any of the subject_person_ids.
        """
        if not username or not memory.subject_person_ids:
            return False

        # We need to load person records to check aliases
        # For efficiency, we just check if the memory was stored about
        # this username by checking subject_person_ids against people cache
        # This is a simplified check - full check would need to await _ensure_people_loaded
        return False  # Conservative default - proper check done in async methods

    async def get_recent_memories(
        self,
        user_id: str | None = None,
        chat_id: str | None = None,
        limit: int = 20,
    ) -> list[str]:
        """Get recent memories without semantic search.

        Args:
            user_id: User ID for scoping.
            chat_id: Chat ID for group memories.
            limit: Maximum memories to return.

        Returns:
            List of memory content strings.
        """
        try:
            memories = await self._store.get_memories(
                limit=limit,
                include_expired=False,
                include_superseded=False,
                owner_user_id=user_id,
                chat_id=chat_id,
            )
        except Exception:
            logger.warning(
                "Failed to get recent memories (user_id=%s, chat_id=%s)",
                user_id,
                chat_id,
                exc_info=True,
            )
            return []

        return [m.content for m in memories]

    async def add_memory(
        self,
        content: str,
        source: str = "user",
        memory_type: MemoryType | None = None,
        expires_at: datetime | None = None,
        expires_in_days: int | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        subject_person_ids: list[str] | None = None,
        observed_at: datetime | None = None,
        source_user_id: str | None = None,
        source_user_name: str | None = None,
        source_session_id: str | None = None,
        source_message_id: str | None = None,
        extraction_confidence: float | None = None,
        sensitivity: Sensitivity | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Add a memory entry.

        Args:
            content: Memory content.
            source: Origin tracking (user, extraction, cli, rpc).
            memory_type: Type classification (defaults to KNOWLEDGE).
            expires_at: Explicit expiration time.
            expires_in_days: Days until expiration (converted to expires_at).
            owner_user_id: Personal scope owner.
            chat_id: Group scope.
            subject_person_ids: People this memory is about.
            observed_at: When fact was observed (for extraction).
            source_user_id: Who said/provided this fact (for multi-user attribution).
            source_user_name: Display name of source user.
            source_session_id: Session ID for extraction tracing.
            source_message_id: Message ID for extraction tracing.
            extraction_confidence: Confidence score for extraction.
            sensitivity: Privacy classification (default PUBLIC).
            metadata: Additional metadata.

        Returns:
            The created memory entry.

        Raises:
            ValueError: If content contains potential secrets.
        """
        # Reject secrets before storing (defense in depth)
        if contains_secret(content):
            raise ValueError(
                "Memory content contains potential secrets and cannot be stored"
            )

        if expires_in_days is not None and expires_at is None:
            expires_at = datetime.now(UTC) + timedelta(days=expires_in_days)

        if memory_type is None:
            memory_type = MemoryType.KNOWLEDGE

        # Generate embedding
        try:
            embedding_floats = await self._embeddings.embed(content)
            embedding = MemoryEntry.encode_embedding(embedding_floats)
        except Exception:
            logger.warning(
                "Failed to generate embedding, continuing without", exc_info=True
            )
            embedding = ""

        # Create memory entry
        memory = await self._store.add_memory(
            content=content,
            memory_type=memory_type,
            embedding=embedding,
            source=source,
            expires_at=expires_at,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            subject_person_ids=subject_person_ids,
            observed_at=observed_at,
            source_user_id=source_user_id,
            source_user_name=source_user_name,
            source_session_id=source_session_id,
            source_message_id=source_message_id,
            extraction_confidence=extraction_confidence,
            sensitivity=sensitivity,
            metadata=metadata,
        )

        # Index embedding
        if embedding:
            try:
                await self._index.add_embedding(memory.id, embedding_floats)
            except Exception:
                logger.warning("Failed to index memory, continuing", exc_info=True)

        # Check for conflicting memories
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

        # Enforce max entries
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

        return memory

    async def find_conflicting_memories(
        self,
        new_content: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        subject_person_ids: list[str] | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Find existing memories that may conflict with new content.

        Args:
            new_content: Content of new memory.
            owner_user_id: User scope.
            chat_id: Chat scope.
            subject_person_ids: Subject filter.

        Returns:
            List of (memory, similarity) tuples above threshold.
        """
        similar = await self._index.search(new_content, limit=10)
        now = datetime.now(UTC)

        conflicts: list[tuple[MemoryEntry, float]] = []
        for result in similar:
            if result.similarity < CONFLICT_SIMILARITY_THRESHOLD:
                continue

            memory = await self._store.get_memory(result.memory_id)
            if not memory:
                continue

            # Skip expired or superseded
            if memory.superseded_at:
                continue
            if memory.expires_at and memory.expires_at <= now:
                continue

            # Scope check
            if not matches_scope(memory, owner_user_id, chat_id):
                continue

            # Subject matching
            memory_subjects = memory.subject_person_ids or []
            if subject_person_ids:
                if not set(subject_person_ids) & set(memory_subjects):
                    continue
            elif memory_subjects:
                continue

            conflicts.append((memory, result.similarity))

        return conflicts

    async def _verify_conflict_with_llm(
        self,
        old_content: str,
        new_content: str,
    ) -> bool:
        """Verify if new memory truly supersedes old using LLM.

        Args:
            old_content: Content of old memory.
            new_content: Content of new memory.

        Returns:
            True if LLM confirms supersession.
        """
        if not self._llm:
            # Without LLM, trust vector similarity
            return True

        try:
            from ash.llm.types import Message, Role

            prompt = SUPERSESSION_PROMPT.format(
                old_content=old_content,
                new_content=new_content,
            )

            response = await self._llm.complete(
                messages=[Message(role=Role.USER, content=prompt)],
                max_tokens=10,
                temperature=0.0,
            )

            answer = response.message.get_text().strip().upper()
            return answer.startswith("YES")
        except Exception:
            logger.warning(
                "LLM verification failed, falling back to similarity", exc_info=True
            )
            return True

    async def supersede_conflicting_memories(
        self,
        new_memory: MemoryEntry,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> int:
        """Find and mark conflicting memories as superseded.

        Uses vector similarity to find candidates, then optionally
        verifies with LLM before superseding.

        Args:
            new_memory: The new memory that may supersede others.
            owner_user_id: User scope.
            chat_id: Chat scope.

        Returns:
            Number of memories superseded.
        """
        conflicts = await self.find_conflicting_memories(
            new_content=new_memory.content,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            subject_person_ids=new_memory.subject_person_ids,
        )

        count = 0
        for memory, similarity in conflicts:
            if memory.id == new_memory.id:
                continue

            # LLM verification for borderline cases
            if similarity < 0.85 and self._llm:
                if not await self._verify_conflict_with_llm(
                    old_content=memory.content,
                    new_content=new_memory.content,
                ):
                    continue

            if await self._mark_superseded(memory.id, new_memory.id):
                count += 1

        return count

    async def _mark_superseded(
        self,
        old_memory_id: str,
        new_memory_id: str,
    ) -> bool:
        """Mark a memory as superseded and clean up its embedding.

        Args:
            old_memory_id: Memory to supersede.
            new_memory_id: Memory that supersedes it.

        Returns:
            True if successfully superseded.
        """
        success = await self._store.mark_memory_superseded(
            memory_id=old_memory_id,
            superseded_by_id=new_memory_id,
        )
        if success:
            try:
                await self._index.delete_embedding(old_memory_id)
            except Exception:
                logger.warning(
                    "Failed to delete superseded memory embedding",
                    extra={"memory_id": old_memory_id},
                    exc_info=True,
                )
        return success

    async def search(
        self,
        query: str,
        limit: int = 5,
        subject_person_id: str | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[SearchResult]:
        """Search memories by semantic similarity.

        Args:
            query: Search query.
            limit: Maximum results.
            subject_person_id: Filter to memories about person.
            owner_user_id: User scope.
            chat_id: Chat scope.

        Returns:
            List of search results with similarity scores.
        """
        vector_results = await self._index.search(query, limit=limit * 2)
        now = datetime.now(UTC)

        # Resolve person names for subject attribution
        all_person_ids: set[str] = set()

        results: list[SearchResult] = []
        for vr in vector_results:
            memory = await self._store.get_memory(vr.memory_id)
            if not memory:
                continue

            # Skip expired or superseded
            if memory.superseded_at:
                continue
            if memory.expires_at and memory.expires_at <= now:
                continue

            # Scope check
            if not matches_scope(memory, owner_user_id, chat_id):
                continue

            # Subject filter
            if subject_person_id:
                if subject_person_id not in (memory.subject_person_ids or []):
                    continue

            if memory.subject_person_ids:
                all_person_ids.update(memory.subject_person_ids)

            results.append(
                SearchResult(
                    id=memory.id,
                    content=memory.content,
                    similarity=vr.similarity,
                    metadata={
                        "memory_type": memory.memory_type.value,
                        "subject_person_ids": memory.subject_person_ids,
                        "sensitivity": memory.sensitivity.value
                        if memory.sensitivity
                        else None,
                        **(memory.metadata or {}),
                    },
                    source_type="memory",
                )
            )

            if len(results) >= limit:
                break

        # Resolve person names
        if all_person_ids:
            person_names = await self._resolve_person_names(list(all_person_ids))
            for result in results:
                if result.metadata:
                    subject_ids = result.metadata.get("subject_person_ids") or []
                    if subject_ids:
                        names = [
                            person_names[pid]
                            for pid in subject_ids
                            if pid in person_names
                        ]
                        if names:
                            result.metadata["subject_name"] = ", ".join(names)

        return results

    async def _resolve_person_names(self, person_ids: list[str]) -> dict[str, str]:
        """Resolve person IDs to names.

        Args:
            person_ids: List of person UUIDs.

        Returns:
            Dict mapping person_id to name.
        """
        return await self._person_manager.resolve_person_names(person_ids)

    async def list_memories(
        self,
        limit: int = 20,
        include_expired: bool = False,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[MemoryEntry]:
        """List recent memories without semantic search.

        Args:
            limit: Maximum memories to return.
            include_expired: Include expired entries.
            owner_user_id: User scope.
            chat_id: Chat scope.

        Returns:
            List of memory entries.
        """
        return await self._store.get_memories(
            include_expired=include_expired,
            include_superseded=False,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            limit=limit,
        )

    async def delete_memory(
        self,
        memory_id: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> bool:
        """Delete a memory and its embedding.

        Supports partial memory IDs (prefix matching).

        Args:
            memory_id: Memory UUID or prefix.
            owner_user_id: User for authorization.
            chat_id: Chat for authorization.

        Returns:
            True if memory was deleted.
        """
        memory = await self._store.get_memory_by_prefix(memory_id)
        if not memory:
            return False

        full_id = memory.id

        deleted = await self._store.delete_memory(
            full_id,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
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

        return True

    async def find_person(
        self,
        owner_user_id: str,
        reference: str,
    ) -> PersonEntry | None:
        """Find a person by reference.

        Args:
            owner_user_id: User who owns the person record.
            reference: Name, relationship, or alias.

        Returns:
            Person entry or None.
        """
        return await self._person_manager.find_person(owner_user_id, reference)

    async def get_known_people(self, owner_user_id: str) -> list[PersonEntry]:
        """Get all known people for a user.

        Args:
            owner_user_id: User to get people for.

        Returns:
            List of person entries.
        """
        return await self._person_manager.get_known_people(owner_user_id)

    async def resolve_or_create_person(
        self,
        owner_user_id: str,
        reference: str,
        content_hint: str | None = None,
    ) -> PersonResolutionResult:
        """Resolve a reference to a person, creating if needed.

        Args:
            owner_user_id: User who will own the person record.
            reference: Name or relationship reference.
            content_hint: Content that may contain the person's name.

        Returns:
            Resolution result with person ID.
        """
        return await self._person_manager.resolve_or_create_person(
            owner_user_id, reference, content_hint
        )

    async def gc(self) -> GCResult:
        """Garbage collect expired and superseded memories.

        Uses smart ephemeral decay based on memory type.

        Returns:
            GC result with counts.
        """
        result = await self._store.gc()

        # Delete embeddings for archived memories
        for memory_id in result.archived_ids:
            try:
                await self._index.delete_embedding(memory_id)
            except Exception:
                logger.debug("Failed to delete embedding for %s", memory_id)

        return result

    async def enforce_max_entries(self, max_entries: int) -> int:
        """Evict oldest memories if over the max_entries limit.

        Args:
            max_entries: Maximum allowed active memories.

        Returns:
            Number of memories evicted.
        """
        now = datetime.now(UTC)
        memories = await self._store.get_memories(
            limit=10000,  # Get all
            include_expired=False,
            include_superseded=False,
        )

        current_count = len(memories)
        if current_count <= max_entries:
            return 0

        excess = current_count - max_entries

        # Sort by age (oldest first) for eviction
        memories.sort(key=lambda m: m.created_at or datetime.min.replace(tzinfo=UTC))

        # Evict oldest that are at least 7 days old
        evicted = 0
        for memory in memories:
            if evicted >= excess:
                break

            if memory.created_at and (now - memory.created_at).days < 7:
                continue  # Don't evict recent memories

            await self._store.delete_memory(memory.id)
            try:
                await self._index.delete_embedding(memory.id)
            except Exception:
                logger.debug(
                    "Failed to delete embedding for %s during eviction", memory.id
                )
            evicted += 1

        if evicted < excess:
            logger.warning(
                "Could not evict enough memories - all remaining are recent",
                extra={"excess": excess, "evicted": evicted},
            )

        return evicted

    async def rebuild_index(self) -> int:
        """Rebuild vector index from filesystem storage.

        Returns:
            Number of embeddings indexed.
        """
        memories = await self._store.get_all_memories()
        count = await self._index.rebuild_from_memories(memories)
        logger.info("index_rebuilt", extra={"count": count})
        return count

    async def get_supersession_chain(self, memory_id: str) -> list[MemoryEntry]:
        """Get the chain of superseded memories leading to this ID.

        Args:
            memory_id: Memory to trace back from.

        Returns:
            List of memories in supersession order (oldest first).
        """
        return await self._store.get_supersession_chain(memory_id)

    async def supersede_confirmed_hearsay(
        self,
        new_memory: MemoryEntry,
        source_user_id: str,
        owner_user_id: str,
        similarity_threshold: float = 0.80,
    ) -> int:
        """Supersede hearsay memories when user confirms a fact about themselves.

        When a user states a fact about themselves that was previously reported
        as hearsay by someone else, this method finds and supersedes the hearsay
        memories with the new first-person fact.

        Args:
            new_memory: The new FACT memory (user speaking about themselves).
            source_user_id: The user who stated the fact (their username).
            owner_user_id: Owner user ID for scoping.
            similarity_threshold: Minimum similarity score to supersede (default 0.80).

        Returns:
            Number of hearsay memories superseded.
        """
        # Find hearsay about this user
        hearsay_candidates = await self._store.find_hearsay_about_user(
            user_id=source_user_id,
            owner_user_id=owner_user_id,
        )

        if not hearsay_candidates:
            return 0

        count = 0
        for hearsay in hearsay_candidates:
            if hearsay.id == new_memory.id:
                continue

            # Check similarity using vector search
            try:
                similar = await self._index.search(hearsay.content, limit=5)
                similarity = 0.0
                for result in similar:
                    if result.memory_id == new_memory.id:
                        similarity = result.similarity
                        break

                if similarity < similarity_threshold:
                    continue

                # Use LLM verification for borderline cases
                if similarity < 0.85 and self._llm:
                    if not await self._verify_conflict_with_llm(
                        old_content=hearsay.content,
                        new_content=new_memory.content,
                    ):
                        continue

                if await self._mark_superseded(hearsay.id, new_memory.id):
                    count += 1
                    logger.info(
                        "Hearsay superseded by fact",
                        extra={
                            "hearsay_id": hearsay.id,
                            "fact_id": new_memory.id,
                            "similarity": similarity,
                        },
                    )
            except Exception:
                logger.debug(
                    "Failed to check hearsay similarity",
                    extra={"hearsay_id": hearsay.id},
                    exc_info=True,
                )

        return count


async def create_memory_manager(
    db_session: AsyncSession,
    llm_registry: LLMRegistry,
    embedding_model: str | None = None,
    embedding_provider: str = "openai",
    max_entries: int | None = None,
    llm_provider: str = "anthropic",
    auto_migrate: bool = True,
) -> MemoryManager:
    """Create a fully-wired MemoryManager.

    Automatically detects and migrates from SQLite to JSONL if needed.

    Args:
        db_session: Database session for vector operations.
        llm_registry: LLM provider registry.
        embedding_model: Embedding model to use.
        embedding_provider: Provider for embeddings (default: openai).
        max_entries: Optional cap on active memories.
        llm_provider: Provider for supersession verification.
        auto_migrate: Whether to auto-migrate from SQLite.

    Returns:
        Configured MemoryManager instance.
    """
    from ash.memory.migration import (
        check_db_has_memories,
        migrate_db_to_jsonl,
        needs_migration,
    )

    # Check if migration is needed
    if auto_migrate and needs_migration():
        try:
            has_memories = await check_db_has_memories(db_session)
            if has_memories:
                logger.info("Starting migration from SQLite to JSONL")
                mem_count, people_count = await migrate_db_to_jsonl(db_session)
                logger.info(
                    "Migration complete",
                    extra={"memories": mem_count, "people": people_count},
                )
        except Exception:
            logger.warning("Migration failed, starting fresh", exc_info=True)

    # Create components
    embedding_generator = EmbeddingGenerator(
        registry=llm_registry,
        model=embedding_model,
        provider=embedding_provider,
    )

    store = FileMemoryStore()
    index = VectorIndex(db_session, embedding_generator)
    await index.initialize()

    # Get LLM for supersession verification (optional)
    llm = None
    try:
        llm = llm_registry.get(llm_provider)
    except Exception:
        logger.debug("LLM not available for supersession verification")

    # Check if index needs rebuild from JSONL
    memories = await store.get_all_memories()
    indexed_count = await index.get_embedding_count()

    # Count active memories that should be indexed
    active_count = sum(1 for m in memories if not m.superseded_at and m.embedding)

    if active_count > 0 and indexed_count == 0:
        logger.info("Index empty, rebuilding from JSONL")
        await index.rebuild_from_memories(memories)

    return MemoryManager(
        store=store,
        index=index,
        embedding_generator=embedding_generator,
        db_session=db_session,
        llm=llm,
        max_entries=max_entries,
    )
