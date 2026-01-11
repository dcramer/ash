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
        max_messages: int = 5,
        max_memories: int = 10,
        min_message_similarity: float = 0.3,
        exclude_message_ids: set[str] | None = None,
    ) -> RetrievedContext:
        """Retrieve relevant context before LLM call.

        Args:
            session_id: Current session ID.
            user_id: User ID for filtering memories. In group chats, this ensures
                User A's memories aren't returned when User B asks a question.
            user_message: The user's message to find relevant context for.
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
            logger.warning("Failed to search messages, continuing without", exc_info=True)

        try:
            # Search memory store - include top N without filtering
            # Filter by owner_user_id to ensure user A's memories aren't shown to user B
            # The retriever already ranks by similarity, so top N are best matches
            memories = await self._retriever.search_memories(
                query=user_message,
                limit=max_memories,
                owner_user_id=user_id,
            )
        except Exception:
            logger.warning("Failed to search memories, continuing without", exc_info=True)

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
            logger.warning("Failed to index messages, continuing without", exc_info=True)

    async def add_memory(
        self,
        content: str,
        source: str = "user",
        expires_at: datetime | None = None,
        expires_in_days: int | None = None,
        owner_user_id: str | None = None,
        subject_person_id: str | None = None,
    ) -> Memory:
        """Add memory entry (used by remember tool).

        Args:
            content: Memory content.
            source: Source of memory (default: "user").
            expires_at: Explicit expiration datetime.
            expires_in_days: Days until expiration (alternative to expires_at).
            owner_user_id: User who added this memory.
            subject_person_id: Person this memory is about.

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
            subject_person_id=subject_person_id,
        )

        # Index for semantic search
        try:
            await self._retriever.index_memory(memory.id, content)
        except Exception:
            logger.warning("Failed to index memory, continuing", exc_info=True)

        return memory

    async def search(
        self,
        query: str,
        limit: int = 5,
        subject_person_id: str | None = None,
        owner_user_id: str | None = None,
    ) -> list[SearchResult]:
        """Search all memory (used by recall tool).

        Args:
            query: Search query.
            limit: Maximum results.
            subject_person_id: Optional filter to memories about a specific person.
            owner_user_id: Optional filter to memories owned by a specific user.

        Returns:
            List of search results sorted by relevance.
        """
        return await self._retriever.search_all(
            query, limit=limit, subject_person_id=subject_person_id,
            owner_user_id=owner_user_id
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
