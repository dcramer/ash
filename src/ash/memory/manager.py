"""Memory manager for orchestrating retrieval and persistence."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ash.db.models import Knowledge, UserProfile
from ash.memory.retrieval import SearchResult, SemanticRetriever
from ash.memory.store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Context retrieved from memory for LLM prompt augmentation."""

    messages: list[SearchResult]
    knowledge: list[SearchResult]
    user_notes: str | None


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
        max_knowledge: int = 3,
    ) -> RetrievedContext:
        """Retrieve relevant context before LLM call.

        Args:
            session_id: Current session ID.
            user_id: User ID for profile lookup.
            user_message: The user's message to find relevant context for.
            max_messages: Maximum number of past messages to retrieve.
            max_knowledge: Maximum number of knowledge entries to retrieve.

        Returns:
            Retrieved context with messages, knowledge, and user notes.
        """
        messages: list[SearchResult] = []
        knowledge: list[SearchResult] = []
        user_notes: str | None = None

        try:
            # Search past messages (across all sessions for this retrieval)
            messages = await self._retriever.search_messages(
                query=user_message,
                limit=max_messages,
            )
        except Exception:
            logger.warning("Failed to search messages, continuing without", exc_info=True)

        try:
            # Search knowledge base
            knowledge = await self._retriever.search_knowledge(
                query=user_message,
                limit=max_knowledge,
            )
        except Exception:
            logger.warning("Failed to search knowledge, continuing without", exc_info=True)

        try:
            # Get user notes
            user_notes = await self.get_user_notes(user_id)
        except Exception:
            logger.warning("Failed to get user notes, continuing without", exc_info=True)

        return RetrievedContext(
            messages=messages,
            knowledge=knowledge,
            user_notes=user_notes,
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

    async def get_user_notes(self, user_id: str) -> str | None:
        """Get user profile notes.

        Args:
            user_id: User ID.

        Returns:
            User notes or None if not found.
        """
        stmt = select(UserProfile.notes).where(UserProfile.user_id == user_id)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return row if row else None

    def format_context_for_prompt(self, context: RetrievedContext) -> str | None:
        """Format retrieved context for inclusion in system prompt.

        Args:
            context: Retrieved context.

        Returns:
            Formatted string or None if no context.
        """
        parts: list[str] = []

        if context.user_notes:
            parts.append(f"## About this user\n{context.user_notes}")

        context_items: list[str] = []
        for item in context.knowledge:
            context_items.append(f"- [Knowledge] {item.content}")
        for item in context.messages:
            context_items.append(f"- [Past conversation] {item.content}")

        if context_items:
            parts.append("## Relevant context from memory\n" + "\n".join(context_items))

        return "\n\n".join(parts) if parts else None
