"""Memory manager for orchestrating retrieval and persistence."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from ash.db.models import Knowledge
from ash.memory.retrieval import SearchResult, SemanticRetriever
from ash.memory.store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Context retrieved from memory for LLM prompt augmentation."""

    messages: list[SearchResult]
    knowledge: list[SearchResult]


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
        max_knowledge: int = 10,
        min_message_similarity: float = 0.3,
    ) -> RetrievedContext:
        """Retrieve relevant context before LLM call.

        Args:
            session_id: Current session ID.
            user_id: User ID (for future use).
            user_message: The user's message to find relevant context for.
            max_messages: Maximum number of past messages to retrieve.
            max_knowledge: Maximum number of knowledge entries to retrieve.
            min_message_similarity: Minimum similarity threshold for messages.
                Knowledge entries are always included (ranked by relevance)
                since a personal assistant typically has a small knowledge base
                where all stored facts are potentially useful.

        Returns:
            Retrieved context with messages and knowledge.
        """
        messages: list[SearchResult] = []
        knowledge: list[SearchResult] = []

        try:
            # Search past messages (across all sessions for this retrieval)
            all_messages = await self._retriever.search_messages(
                query=user_message,
                limit=max_messages,
            )
            # Filter messages by similarity threshold (they can be noisy)
            messages = [m for m in all_messages if m.similarity >= min_message_similarity]
        except Exception:
            logger.warning("Failed to search messages, continuing without", exc_info=True)

        try:
            # Search knowledge base - include top N without filtering
            # For a personal assistant, stored facts are always relevant
            # The retriever already ranks by similarity, so top N are best matches
            knowledge = await self._retriever.search_knowledge(
                query=user_message,
                limit=max_knowledge,
            )
        except Exception:
            logger.warning("Failed to search knowledge, continuing without", exc_info=True)

        return RetrievedContext(
            messages=messages,
            knowledge=knowledge,
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
        # Store messages
        user_msg = await self._store.add_message(
            session_id=session_id,
            role="user",
            content=user_message,
        )

        assistant_msg = await self._store.add_message(
            session_id=session_id,
            role="assistant",
            content=assistant_response,
        )

        # Index for semantic search
        try:
            await self._retriever.index_message(user_msg.id, user_message)
            await self._retriever.index_message(assistant_msg.id, assistant_response)
        except Exception:
            logger.warning("Failed to index messages, continuing without", exc_info=True)

    async def add_knowledge(
        self,
        content: str,
        source: str = "user",
        expires_at: datetime | None = None,
        expires_in_days: int | None = None,
    ) -> Knowledge:
        """Add knowledge entry (used by remember tool).

        Args:
            content: Knowledge content.
            source: Source of knowledge (default: "user").
            expires_at: Explicit expiration datetime.
            expires_in_days: Days until expiration (alternative to expires_at).

        Returns:
            Created knowledge entry.
        """
        # Calculate expiration if days provided
        if expires_in_days is not None and expires_at is None:
            expires_at = datetime.now(UTC) + timedelta(days=expires_in_days)

        # Store knowledge
        knowledge = await self._store.add_knowledge(
            content=content,
            source=source,
            expires_at=expires_at,
        )

        # Index for semantic search
        try:
            await self._retriever.index_knowledge(knowledge.id, content)
        except Exception:
            logger.warning("Failed to index knowledge, continuing", exc_info=True)

        return knowledge

    async def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search all memory (used by recall tool).

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of search results sorted by relevance.
        """
        return await self._retriever.search_all(query, limit=limit)
