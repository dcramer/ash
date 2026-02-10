"""Memory manager for orchestrating retrieval and persistence.

This module provides the primary facade for all memory operations,
using filesystem-primary storage with a SQLite vector index.
"""

from __future__ import annotations

import logging
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
    matches_scope,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from ash.llm import LLMProvider, LLMRegistry

logger = logging.getLogger(__name__)

CONFLICT_SIMILARITY_THRESHOLD = 0.75

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
    ) -> RetrievedContext:
        """Retrieve relevant memory context before LLM call.

        Args:
            user_id: User ID for scoping.
            user_message: Message to find relevant context for.
            chat_id: Optional chat ID for group memories.
            max_memories: Maximum memories to return.

        Returns:
            Retrieved context with matching memories.
        """
        try:
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

        return RetrievedContext(memories=memories)

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
            metadata: Additional metadata.

        Returns:
            The created memory entry.
        """
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

            success = await self._store.mark_memory_superseded(
                memory_id=memory.id,
                superseded_by_id=new_memory.id,
            )
            if success:
                try:
                    await self._index.delete_embedding(memory.id)
                except Exception:
                    logger.warning(
                        "Failed to delete superseded memory embedding",
                        extra={"memory_id": memory.id},
                        exc_info=True,
                    )
                count += 1

        return count

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
