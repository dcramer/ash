"""Memory manager for orchestrating retrieval and persistence.

Note: Message/conversation persistence has been moved to ash.sessions module.
This module now only handles memory operations (facts, relationships).
"""

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from ash.db.models import Memory, Person
from ash.memory.retrieval import SemanticRetriever
from ash.memory.store import MemoryStore
from ash.memory.types import PersonResolutionResult, RetrievedContext, SearchResult

if TYPE_CHECKING:
    from ash.llm import LLMRegistry

logger = logging.getLogger(__name__)


# Similarity threshold for detecting conflicting memories
# Higher threshold = stricter matching, fewer false positives
CONFLICT_SIMILARITY_THRESHOLD = 0.75

# Known relationship terms for parsing references
RELATIONSHIP_TERMS = {
    "wife",
    "husband",
    "partner",
    "spouse",
    "mom",
    "mother",
    "dad",
    "father",
    "parent",
    "son",
    "daughter",
    "child",
    "kid",
    "brother",
    "sister",
    "sibling",
    "boss",
    "manager",
    "coworker",
    "colleague",
    "friend",
    "best friend",
    "roommate",
    "doctor",
    "therapist",
    "dentist",
}


class MemoryManager:
    """Orchestrates memory retrieval and persistence.

    This class coordinates between MemoryStore (data access) and
    SemanticRetriever (vector search) to provide a unified interface
    for the agent's memory operations.
    """

    def __init__(
        self,
        store: MemoryStore,
        retriever: SemanticRetriever,
        db_session: AsyncSession,
        max_entries: int | None = None,
    ):
        """Initialize memory manager.

        Args:
            store: Memory store for data access.
            retriever: Semantic retriever for vector search.
            db_session: Database session for direct queries.
            max_entries: Optional cap on active memories (None = unlimited).
        """
        self._store = store
        self._retriever = retriever
        self._session = db_session
        self._max_entries = max_entries

    async def get_context_for_message(
        self,
        user_id: str,
        user_message: str,
        chat_id: str | None = None,
        max_memories: int = 10,
    ) -> RetrievedContext:
        """Retrieve relevant memory context before LLM call.

        Memory scoping:
        - Personal: user_id set - only that user's memories
        - Group: chat_id set - include group memories for that chat

        Args:
            user_id: User ID for filtering personal memories.
            user_message: The user's message to find relevant context for.
            chat_id: Chat ID for filtering group memories.
            max_memories: Maximum number of memory entries to retrieve.

        Returns:
            Retrieved context with relevant memories.
        """
        memories: list[SearchResult] = []

        try:
            # Search memory store - include top N
            # Filter by owner_user_id for personal memories and chat_id for group memories
            # The retriever already ranks by similarity, so top N are best matches
            memories = await self._retriever.search_memories(
                query=user_message,
                limit=max_memories,
                owner_user_id=user_id,
                chat_id=chat_id,
            )
        except Exception:
            logger.warning(
                "Failed to search memories, continuing without", exc_info=True
            )

        return RetrievedContext(memories=memories)

    async def get_recent_memories(
        self,
        user_id: str | None = None,
        chat_id: str | None = None,
        limit: int = 20,
    ) -> list[str]:
        """Get recent memories without semantic search.

        Used when we need to know what memories exist without a specific query,
        e.g., for memory extraction to avoid duplicates.

        Args:
            user_id: Filter to user's personal memories.
            chat_id: Filter to include group memories for this chat.
            limit: Maximum number of memories to return.

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
        expires_at: datetime | None = None,
        expires_in_days: int | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        subject_person_ids: list[str] | None = None,
    ) -> Memory:
        """Add memory entry (used by remember tool).

        Memory scoping:
        - Personal: owner_user_id set, chat_id NULL - only visible to that user
        - Group: owner_user_id NULL, chat_id set - visible to everyone in that chat

        Args:
            content: Memory content.
            source: Source of memory (default: "user").
            expires_at: Explicit expiration datetime.
            expires_in_days: Days until expiration (alternative to expires_at).
            owner_user_id: User who added this memory (NULL for group memories).
            chat_id: Chat this memory belongs to (NULL for personal memories).
            subject_person_ids: List of person IDs this memory is about.

        Returns:
            Created memory entry.
        """
        # Calculate expiration if days provided
        if expires_in_days is not None and expires_at is None:
            expires_at = datetime.now(UTC) + timedelta(days=expires_in_days)

        # Store memory
        memory = await self._store.add_memory(
            content=content,
            source=source,
            expires_at=expires_at,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            subject_person_ids=subject_person_ids,
        )

        # Index for semantic search
        try:
            await self._retriever.index_memory(memory.id, content)
        except Exception:
            logger.warning("Failed to index memory, continuing", exc_info=True)

        # Check for and supersede conflicting memories
        try:
            superseded_count = await self.supersede_conflicting_memories(
                new_memory_id=memory.id,
                new_content=content,
                owner_user_id=owner_user_id,
                chat_id=chat_id,
                subject_person_ids=subject_person_ids,
            )
            if superseded_count > 0:
                logger.debug(
                    "Memory superseded %d older entries",
                    superseded_count,
                    extra={"new_memory_id": memory.id},
                )
        except Exception:
            logger.warning("Failed to check for conflicting memories", exc_info=True)

        # Enforce max_entries limit if configured
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
    ) -> list[tuple[str, float]]:
        """Find existing memories that may conflict with new content.

        Looks for memories with high semantic similarity in the same scope,
        which likely represent updated information about the same topic.

        Args:
            new_content: The new memory content to check against.
            owner_user_id: Scope to user's personal memories.
            chat_id: Scope to group memories.
            subject_person_ids: Filter to memories with overlapping subjects.

        Returns:
            List of (memory_id, similarity_score) tuples for potential conflicts.
        """
        # Search for similar memories in the same scope
        similar_memories = await self._retriever.search_memories(
            query=new_content,
            limit=10,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            include_expired=False,
            include_superseded=False,
        )

        conflicts = []
        for result in similar_memories:
            # Check similarity threshold
            if result.similarity < CONFLICT_SIMILARITY_THRESHOLD:
                continue

            # Get subjects from the existing memory
            result_subjects = (
                result.metadata.get("subject_person_ids") if result.metadata else None
            ) or []

            # Subject matching rules:
            # - If NEW has subjects: OLD must have overlapping subjects
            # - If NEW has no subjects: OLD must also have no subjects
            # This prevents general facts from superseding person-specific facts
            if subject_person_ids:
                # New memory has subjects - require overlap
                if not set(subject_person_ids) & set(result_subjects):
                    continue
            else:
                # New memory has no subjects - only conflict with other no-subject memories
                if result_subjects:
                    continue

            conflicts.append((result.id, result.similarity))

        return conflicts

    async def supersede_conflicting_memories(
        self,
        new_memory_id: str,
        new_content: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        subject_person_ids: list[str] | None = None,
    ) -> int:
        """Find and mark conflicting memories as superseded.

        Called after a new memory is added to check for and handle conflicts.
        Superseded memories have their embeddings deleted to prevent
        storage bloat and potential visibility issues.

        Args:
            new_memory_id: ID of the newly added memory.
            new_content: Content of the new memory.
            owner_user_id: Scope to user's personal memories.
            chat_id: Scope to group memories.
            subject_person_ids: Subjects the memory is about.

        Returns:
            Number of memories marked as superseded.
        """
        conflicts = await self.find_conflicting_memories(
            new_content=new_content,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            subject_person_ids=subject_person_ids,
        )

        count = 0
        for memory_id, _similarity in conflicts:
            # Don't supersede the new memory itself
            if memory_id == new_memory_id:
                continue

            success = await self._store.mark_memory_superseded(
                memory_id=memory_id,
                superseded_by_id=new_memory_id,
            )
            if success:
                # Clean up the embedding for the superseded memory
                try:
                    await self._retriever.delete_memory_embedding(memory_id)
                except Exception:
                    logger.warning(
                        "Failed to delete superseded memory embedding",
                        extra={"memory_id": memory_id},
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
        """Search memory (used by recall tool).

        Args:
            query: Search query.
            limit: Maximum results.
            subject_person_id: Optional filter to memories about a specific person.
            owner_user_id: Filter to user's personal memories.
            chat_id: Filter to include group memories for this chat.

        Returns:
            List of search results sorted by relevance.
        """
        return await self._retriever.search(
            query,
            limit=limit,
            subject_person_id=subject_person_id,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
        )

    async def list_memories(
        self,
        limit: int = 20,
        include_expired: bool = False,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[Memory]:
        """List recent memories without semantic search.

        Args:
            limit: Maximum results.
            include_expired: Include expired entries.
            owner_user_id: Filter to user's personal memories.
            chat_id: Filter to include group memories for this chat.

        Returns:
            List of recent memories.
        """
        return await self._store.get_memories(
            include_expired=include_expired,
            include_superseded=False,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            limit=limit,
        )

    # Person operations

    async def find_person(
        self,
        owner_user_id: str,
        reference: str,
    ) -> Person | None:
        """Find a person by reference (for recall tool).

        Args:
            owner_user_id: User who owns this person reference.
            reference: Name, relationship, or alias.

        Returns:
            Person if found, None otherwise.
        """
        return await self._store.find_person_by_reference(owner_user_id, reference)

    async def get_known_people(self, owner_user_id: str) -> list[Person]:
        """Get all known people for a user (for prompt context).

        Args:
            owner_user_id: User ID.

        Returns:
            List of people.
        """
        return await self._store.get_people_for_user(owner_user_id)

    async def resolve_or_create_person(
        self,
        owner_user_id: str,
        reference: str,
        content_hint: str | None = None,
    ) -> PersonResolutionResult:
        """Resolve a reference to a person, creating if needed.

        Args:
            owner_user_id: User who owns this person reference.
            reference: How user referred to the person ("my wife", "Sarah", "boss").
            content_hint: The content being stored, may contain the person's name.

        Returns:
            PersonResolutionResult with person_id and whether it was created.
        """
        # Try to find existing person
        existing = await self._store.find_person_by_reference(owner_user_id, reference)
        if existing:
            return PersonResolutionResult(
                person_id=existing.id,
                created=False,
                person_name=existing.name,
            )

        # Need to create - determine name and relationship
        name, relationship = self._parse_person_reference(reference, content_hint)

        person = await self._store.create_person(
            owner_user_id=owner_user_id,
            name=name,
            relationship=relationship,
            aliases=[reference] if reference.lower() != name.lower() else None,
        )

        return PersonResolutionResult(
            person_id=person.id,
            created=True,
            person_name=person.name,
        )

    def _parse_person_reference(
        self,
        reference: str,
        content_hint: str | None = None,
    ) -> tuple[str, str | None]:
        """Parse a person reference into name and relationship.

        Args:
            reference: How user referred to the person.
            content_hint: Content that might contain the actual name.

        Returns:
            Tuple of (name, relationship).
        """
        ref_lower = reference.lower().strip()

        # Strip @ prefix for usernames (e.g., "@notzeeg" -> "notzeeg")
        if ref_lower.startswith("@"):
            ref_lower = ref_lower[1:]

        # Remove "my " prefix if present
        relationship = ref_lower[3:] if ref_lower.startswith("my ") else None

        # If reference is a relationship term, try to extract name from content
        if relationship and relationship in RELATIONSHIP_TERMS:
            if content_hint:
                # Try to extract a name from content
                name = self._extract_name_from_content(content_hint, relationship)
                if name:
                    return name, relationship
            # Use capitalized relationship as placeholder name
            return relationship.title(), relationship

        # Reference is likely a name
        return ref_lower.title(), relationship

    def _extract_name_from_content(
        self,
        content: str,
        relationship: str,
    ) -> str | None:
        """Try to extract a person's name from content.

        Looks for patterns like:
        - "Sarah's birthday is..."
        - "wife's name is Sarah"
        - "My wife Sarah likes..."
        - "wife is named Sarah"

        Limitations:
        - Only extracts single-word names (misses "Mary Jane", "Dr. Smith")
        - Conservative matching to avoid false positives
        - Names must start with a letter and contain only word characters
        """
        # Pattern: "X's name is Y" or "X is named Y"
        name_is_pattern = rf"{relationship}(?:'s name is| is named) (\w+)"
        match = re.search(name_is_pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern: "My [relationship] [Name]" at start or after comma
        my_pattern = rf"(?:^|,\s*)my {relationship} (\w+)"
        match = re.search(my_pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern: "[Name]'s" at the start (possessive name)
        possessive_pattern = r"^(\w+)'s\s"
        match = re.search(possessive_pattern, content)
        if match:
            name = match.group(1)
            # Avoid false positives like "User's"
            if name.lower() not in ["user", "my", "the", "their", "his", "her"]:
                return name

        return None

    async def _delete_memories_with_embeddings(self, memory_ids: list[str]) -> None:
        """Delete memories and their embeddings.

        Args:
            memory_ids: List of memory IDs to delete.
        """
        from sqlalchemy import delete, text

        if not memory_ids:
            return

        # Delete embeddings first
        for memory_id in memory_ids:
            try:
                await self._session.execute(
                    text("DELETE FROM memory_embeddings WHERE memory_id = :id"),
                    {"id": memory_id},
                )
            except Exception:
                logger.debug("Failed to delete embedding for %s", memory_id)

        # Delete the memories
        await self._session.execute(delete(Memory).where(Memory.id.in_(memory_ids)))

    async def gc(self) -> tuple[int, int]:
        """Garbage collect expired and superseded memories.

        Removes memories that are either:
        - Past their expires_at date
        - Marked as superseded by another memory

        Also cleans up associated embeddings.

        Returns:
            Tuple of (expired_count, superseded_count) deleted.
        """
        from sqlalchemy import select

        now = datetime.now(UTC)

        # Find expired memories
        expired_stmt = select(Memory.id).where(Memory.expires_at <= now)
        expired_result = await self._session.execute(expired_stmt)
        expired_ids = [r[0] for r in expired_result.all()]

        # Find superseded memories
        superseded_stmt = select(Memory.id).where(Memory.superseded_at.isnot(None))
        superseded_result = await self._session.execute(superseded_stmt)
        superseded_ids = [r[0] for r in superseded_result.all()]

        # Combine and deduplicate
        all_ids = list(set(expired_ids) | set(superseded_ids))

        if not all_ids:
            return (0, 0)

        await self._delete_memories_with_embeddings(all_ids)
        await self._session.commit()

        logger.info(
            "Memory garbage collection complete",
            extra={
                "expired_count": len(expired_ids),
                "superseded_count": len(superseded_ids),
            },
        )

        return (len(expired_ids), len(superseded_ids))

    async def enforce_max_entries(self, max_entries: int) -> int:
        """Evict oldest memories if over the max_entries limit.

        Eviction priority:
        1. Superseded memories (oldest first)
        2. Expired memories (oldest first)
        3. Active memories older than 7 days (oldest first)

        Args:
            max_entries: Maximum number of active memories to keep.

        Returns:
            Number of memories evicted.
        """
        from sqlalchemy import func, select

        # Count active (non-superseded, non-expired) memories
        now = datetime.now(UTC)
        count_stmt = (
            select(func.count(Memory.id))
            .where(Memory.superseded_at.is_(None))
            .where((Memory.expires_at.is_(None)) | (Memory.expires_at > now))
        )
        result = await self._session.execute(count_stmt)
        current_count = result.scalar() or 0

        if current_count <= max_entries:
            return 0

        excess = current_count - max_entries
        evicted = 0

        # Build eviction queries in priority order
        eviction_queries = [
            # First: superseded memories (oldest first)
            select(Memory.id)
            .where(Memory.superseded_at.isnot(None))
            .order_by(Memory.created_at.asc()),
            # Second: expired memories (oldest first)
            select(Memory.id)
            .where(Memory.expires_at <= now)
            .order_by(Memory.created_at.asc()),
            # Third: active memories older than 7 days (oldest first)
            select(Memory.id)
            .where(Memory.superseded_at.is_(None))
            .where((Memory.expires_at.is_(None)) | (Memory.expires_at > now))
            .where(Memory.created_at < now - timedelta(days=7))
            .order_by(Memory.created_at.asc()),
        ]

        for query in eviction_queries:
            if evicted >= excess:
                break

            result = await self._session.execute(query.limit(excess - evicted))
            ids_to_evict = [r[0] for r in result.all()]

            if ids_to_evict:
                await self._delete_memories_with_embeddings(ids_to_evict)
                evicted += len(ids_to_evict)

        await self._session.commit()

        if evicted < excess:
            logger.warning(
                "Could not evict enough memories - all remaining are recent",
                extra={"excess": excess, "evicted": evicted},
            )

        return evicted


async def create_memory_manager(
    db_session: AsyncSession,
    llm_registry: "LLMRegistry",
    embedding_model: str | None = None,
    embedding_provider: str = "openai",
    max_entries: int | None = None,
) -> MemoryManager:
    """Create a fully-wired MemoryManager.

    This factory handles the wiring of internal components:
    - EmbeddingGenerator for vector embeddings
    - SemanticRetriever for vector search
    - MemoryStore for data access

    Args:
        db_session: Database session for persistence.
        llm_registry: LLM registry for embedding provider.
        embedding_model: Embedding model name (default: provider default).
        embedding_provider: Embedding provider (default: "openai").
        max_entries: Optional cap on active memories.

    Returns:
        Configured MemoryManager.
    """
    from ash.memory.embeddings import EmbeddingGenerator
    from ash.memory.retrieval import SemanticRetriever
    from ash.memory.store import MemoryStore

    embedding_generator = EmbeddingGenerator(
        registry=llm_registry,
        model=embedding_model,
        provider=embedding_provider,
    )

    store = MemoryStore(db_session)
    retriever = SemanticRetriever(db_session, embedding_generator)
    await retriever.initialize_vector_tables()

    return MemoryManager(
        store=store,
        retriever=retriever,
        db_session=db_session,
        max_entries=max_entries,
    )
