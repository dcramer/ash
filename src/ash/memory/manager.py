"""Memory manager for orchestrating retrieval and persistence."""

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from ash.db.models import Memory, Person
from ash.memory.retrieval import SearchResult, SemanticRetriever
from ash.memory.store import MemoryStore

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


@dataclass
class PersonResolutionResult:
    """Result of person resolution."""

    person_id: str
    created: bool
    person_name: str


@dataclass
class RetrievedContext:
    """Context retrieved from memory for LLM prompt augmentation."""

    messages: list[SearchResult]
    memories: list[SearchResult]


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
    ):
        """Initialize memory manager.

        Args:
            store: Memory store for data access.
            retriever: Semantic retriever for vector search.
            db_session: Database session for direct queries.
        """
        self._store = store
        self._retriever = retriever
        self._session = db_session

    async def get_context_for_message(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        chat_id: str | None = None,
        max_messages: int = 5,
        max_memories: int = 10,
        min_message_similarity: float = 0.3,
        exclude_message_ids: set[str] | None = None,
    ) -> RetrievedContext:
        """Retrieve relevant context before LLM call.

        Memory scoping:
        - Personal: user_id set - only that user's memories
        - Group: chat_id set - include group memories for that chat

        Args:
            session_id: Current session ID.
            user_id: User ID for filtering personal memories.
            user_message: The user's message to find relevant context for.
            chat_id: Chat ID for filtering group memories.
            max_messages: Maximum number of past messages to retrieve.
            max_memories: Maximum number of memory entries to retrieve.
            min_message_similarity: Minimum similarity threshold for messages.
                Memory entries are always included (ranked by relevance)
                since a personal assistant typically has a small memory store
                where all stored facts are potentially useful.
            exclude_message_ids: Message IDs to exclude (e.g., already in context).

        Returns:
            Retrieved context with messages and memories.
        """
        messages: list[SearchResult] = []
        memories: list[SearchResult] = []

        try:
            # Search past messages (across all sessions for this retrieval)
            # Request extra results to account for exclusions
            extra = len(exclude_message_ids) if exclude_message_ids else 0
            all_messages = await self._retriever.search_messages(
                query=user_message,
                limit=max_messages + extra,
            )
            # Filter by similarity threshold AND exclude duplicates
            for m in all_messages:
                if m.similarity >= min_message_similarity:
                    if exclude_message_ids and m.id in exclude_message_ids:
                        continue
                    messages.append(m)
                    if len(messages) >= max_messages:
                        break
        except Exception:
            logger.warning(
                "Failed to search messages, continuing without", exc_info=True
            )

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

        return RetrievedContext(
            messages=messages,
            memories=memories,
        )

    async def persist_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """Store and index a conversation turn.

        Args:
            session_id: Session ID.
            user_message: User's message.
            assistant_response: Assistant's response.
        """
        from ash.core.tokens import estimate_tokens

        # Store messages with token estimates
        user_msg = await self._store.add_message(
            session_id=session_id,
            role="user",
            content=user_message,
            token_count=estimate_tokens(user_message),
        )

        assistant_msg = await self._store.add_message(
            session_id=session_id,
            role="assistant",
            content=assistant_response,
            token_count=estimate_tokens(assistant_response),
        )

        # Index for semantic search
        try:
            await self._retriever.index_message(user_msg.id, user_message)
            await self._retriever.index_message(assistant_msg.id, assistant_response)
        except Exception:
            logger.warning(
                "Failed to index messages, continuing without", exc_info=True
            )

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
                logger.info(
                    "Memory superseded older entries",
                    extra={
                        "new_memory_id": memory.id,
                        "superseded_count": superseded_count,
                    },
                )
        except Exception:
            logger.warning("Failed to check for conflicting memories", exc_info=True)

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
        for memory_id, similarity in conflicts:
            # Don't supersede the new memory itself
            if memory_id == new_memory_id:
                continue

            success = await self._store.mark_memory_superseded(
                memory_id=memory_id,
                superseded_by_id=new_memory_id,
            )
            if success:
                count += 1
                logger.info(
                    "Superseded memory",
                    extra={
                        "memory_id": memory_id,
                        "superseded_by": new_memory_id,
                        "similarity": similarity,
                    },
                )

        return count

    async def search(
        self,
        query: str,
        limit: int = 5,
        subject_person_id: str | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[SearchResult]:
        """Search all memory (used by recall tool).

        Args:
            query: Search query.
            limit: Maximum results.
            subject_person_id: Optional filter to memories about a specific person.
            owner_user_id: Filter to user's personal memories.
            chat_id: Filter to include group memories for this chat.

        Returns:
            List of search results sorted by relevance.
        """
        return await self._retriever.search_all(
            query,
            limit=limit,
            subject_person_id=subject_person_id,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
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

        # Remove "my " prefix if present
        relationship: str | None = None
        if ref_lower.startswith("my "):
            relationship = ref_lower[3:]  # "wife", "boss", etc.
        else:
            relationship = None

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
        return reference.title(), relationship

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
        """
        # Pattern: "X's name is Y"
        name_is_pattern = rf"{relationship}'s name is (\w+)"
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
