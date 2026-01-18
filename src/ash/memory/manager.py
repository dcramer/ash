"""Memory manager for orchestrating retrieval and persistence."""

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

CONFLICT_SIMILARITY_THRESHOLD = 0.75

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
    """Orchestrates memory retrieval and persistence."""

    def __init__(
        self,
        store: MemoryStore,
        retriever: SemanticRetriever,
        db_session: AsyncSession,
        max_entries: int | None = None,
    ) -> None:
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
        """Retrieve relevant memory context before LLM call."""
        try:
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
            memories = []

        return RetrievedContext(memories=memories)

    async def get_recent_memories(
        self,
        user_id: str | None = None,
        chat_id: str | None = None,
        limit: int = 20,
    ) -> list[str]:
        """Get recent memories without semantic search."""
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
        """Add memory entry (used by remember tool)."""
        if expires_in_days is not None and expires_at is None:
            expires_at = datetime.now(UTC) + timedelta(days=expires_in_days)

        memory = await self._store.add_memory(
            content=content,
            source=source,
            expires_at=expires_at,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            subject_person_ids=subject_person_ids,
        )

        try:
            await self._retriever.index_memory(memory.id, content)
        except Exception:
            logger.warning("Failed to index memory, continuing", exc_info=True)

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
        """Find existing memories that may conflict with new content."""
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
            if result.similarity < CONFLICT_SIMILARITY_THRESHOLD:
                continue

            result_subjects = (
                result.metadata.get("subject_person_ids") if result.metadata else None
            ) or []

            # Subject matching: new with subjects requires overlap, new without subjects
            # only conflicts with other no-subject memories
            if subject_person_ids:
                if not set(subject_person_ids) & set(result_subjects):
                    continue
            elif result_subjects:
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
        """Find and mark conflicting memories as superseded."""
        conflicts = await self.find_conflicting_memories(
            new_content=new_content,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            subject_person_ids=subject_person_ids,
        )

        count = 0
        for memory_id, _ in conflicts:
            if memory_id == new_memory_id:
                continue

            success = await self._store.mark_memory_superseded(
                memory_id=memory_id,
                superseded_by_id=new_memory_id,
            )
            if success:
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
        """Search memory (used by recall tool)."""
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
        """List recent memories without semantic search."""
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
        """
        # Look up full memory ID first (supports prefix matching)
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
            await self._retriever.delete_memory_embedding(full_id)
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
    ) -> Person | None:
        """Find a person by reference (for recall tool)."""
        return await self._store.find_person_by_reference(owner_user_id, reference)

    async def get_known_people(self, owner_user_id: str) -> list[Person]:
        """Get all known people for a user (for prompt context)."""
        return await self._store.get_people_for_user(owner_user_id)

    async def resolve_or_create_person(
        self,
        owner_user_id: str,
        reference: str,
        content_hint: str | None = None,
    ) -> PersonResolutionResult:
        """Resolve a reference to a person, creating if needed."""
        existing = await self._store.find_person_by_reference(owner_user_id, reference)
        if existing:
            return PersonResolutionResult(
                person_id=existing.id,
                created=False,
                person_name=existing.name,
            )

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
        """Parse a person reference into name and relationship."""
        ref_lower = reference.lower().strip()

        if ref_lower.startswith("@"):
            ref_lower = ref_lower[1:]

        relationship = ref_lower[3:] if ref_lower.startswith("my ") else None

        if relationship and relationship in RELATIONSHIP_TERMS:
            if content_hint:
                name = self._extract_name_from_content(content_hint, relationship)
                if name:
                    return name, relationship
            return relationship.title(), relationship

        return ref_lower.title(), relationship

    def _extract_name_from_content(
        self,
        content: str,
        relationship: str,
    ) -> str | None:
        """Try to extract a person's name from content."""
        # Pattern: "X's name is Y" or "X is named Y"
        match = re.search(
            rf"{relationship}(?:'s name is| is named) (\w+)", content, re.IGNORECASE
        )
        if match:
            return match.group(1)

        # Pattern: "My [relationship] [Name]" at start or after comma
        match = re.search(rf"(?:^|,\s*)my {relationship} (\w+)", content, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern: "[Name]'s" at the start (possessive name)
        match = re.search(r"^(\w+)'s\s", content)
        if match:
            name = match.group(1)
            if name.lower() not in ["user", "my", "the", "their", "his", "her"]:
                return name

        return None

    async def _delete_memories_with_embeddings(self, memory_ids: list[str]) -> None:
        """Delete memories and their embeddings."""
        from sqlalchemy import delete, text

        if not memory_ids:
            return

        for memory_id in memory_ids:
            try:
                await self._session.execute(
                    text("DELETE FROM memory_embeddings WHERE memory_id = :id"),
                    {"id": memory_id},
                )
            except Exception:
                logger.debug("Failed to delete embedding for %s", memory_id)

        await self._session.execute(delete(Memory).where(Memory.id.in_(memory_ids)))

    async def gc(self) -> tuple[int, int]:
        """Garbage collect expired and superseded memories."""
        from sqlalchemy import select

        now = datetime.now(UTC)

        expired_result = await self._session.execute(
            select(Memory.id).where(Memory.expires_at <= now)
        )
        expired_ids = [r[0] for r in expired_result.all()]

        superseded_result = await self._session.execute(
            select(Memory.id).where(Memory.superseded_at.isnot(None))
        )
        superseded_ids = [r[0] for r in superseded_result.all()]

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
        """Evict oldest memories if over the max_entries limit."""
        from sqlalchemy import func, select

        now = datetime.now(UTC)
        result = await self._session.execute(
            select(func.count(Memory.id))
            .where(Memory.superseded_at.is_(None))
            .where((Memory.expires_at.is_(None)) | (Memory.expires_at > now))
        )
        current_count = result.scalar() or 0

        if current_count <= max_entries:
            return 0

        excess = current_count - max_entries
        evicted = 0

        # Eviction priority: superseded, expired, then old active memories
        eviction_queries = [
            select(Memory.id)
            .where(Memory.superseded_at.isnot(None))
            .order_by(Memory.created_at.asc()),
            select(Memory.id)
            .where(Memory.expires_at <= now)
            .order_by(Memory.created_at.asc()),
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
    """Create a fully-wired MemoryManager."""
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
