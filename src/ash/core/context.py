"""Context gathering for agent message processing.

Extracts context retrieval logic from the Agent into a single-responsibility
class that handles memory lookup, participant resolution, and context assembly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ash.store.store import Store
    from ash.store.types import PersonEntry, RetrievedContext


logger = logging.getLogger(__name__)


@dataclass
class GatheredContext:
    """Context gathered for message processing.

    Contains all retrieved memories, known people, and participant info
    needed to build the system prompt.
    """

    memory: RetrievedContext | None = None
    known_people: list[PersonEntry] | None = None
    sender_person: PersonEntry | None = None
    participant_person_ids: dict[str, set[str]] | None = None


class ContextGatherer:
    """Gathers all context needed for processing a message.

    Single responsibility: retrieve memories, resolve participants,
    and assemble context for prompt building. Extracted from Agent
    to reduce complexity and improve testability.
    """

    def __init__(self, store: Store | None):
        """Initialize context gatherer.

        Args:
            store: Unified store for memory and people operations.
                   If None, context gathering is disabled.
        """
        self._store = store

    async def gather(
        self,
        user_id: str | None,
        user_message: str,
        chat_id: str | None = None,
        chat_type: str | None = None,
        sender_username: str | None = None,
    ) -> GatheredContext:
        """Gather all context for a message.

        Args:
            user_id: The effective user ID for the message.
            user_message: The user's message text.
            chat_id: Optional chat ID for scoping.
            chat_type: Type of chat ("group", "supergroup", "private").
            sender_username: Username of the message sender.

        Returns:
            GatheredContext with retrieved memories and known people.
        """
        if not self._store:
            return GatheredContext()

        sender_person_ids = await self._resolve_sender_person_ids(sender_username)

        memory_context = await self._retrieve_memories(
            user_id=user_id,
            user_message=user_message,
            chat_id=chat_id,
            chat_type=chat_type,
            sender_username=sender_username,
            sender_person_ids=sender_person_ids,
        )

        known_people = await self._list_known_people(user_id)
        sender_person = await self._resolve_sender_person(
            sender_username,
            person_ids=sender_person_ids,
        )

        return GatheredContext(
            memory=memory_context,
            known_people=known_people,
            sender_person=sender_person,
            participant_person_ids=None,  # Set during memory retrieval
        )

    async def _retrieve_memories(
        self,
        user_id: str | None,
        user_message: str,
        chat_id: str | None = None,
        chat_type: str | None = None,
        sender_username: str | None = None,
        sender_person_ids: set[str] | None = None,
    ) -> RetrievedContext | None:
        """Retrieve memory context for a message.

        Resolves sender's person IDs for cross-context retrieval,
        then fetches relevant memories.
        """
        if not self._store or not user_id:
            return None

        try:
            start_time = time.monotonic()

            # Build participant info for cross-context retrieval
            participant_person_ids: dict[str, set[str]] | None = None

            # Resolve sender's person IDs for cross-context (both private and group).
            # Privacy filter already blocks SENSITIVE facts in group chats.
            if sender_username:
                resolved_person_ids = sender_person_ids
                if resolved_person_ids is None:
                    resolved_person_ids = await self._resolve_sender_person_ids(
                        sender_username
                    )
                try:
                    if resolved_person_ids:
                        participant_person_ids = {sender_username: resolved_person_ids}
                except Exception:
                    logger.debug("Failed to resolve participant person IDs")

            memory_context = await self._store.get_context_for_message(
                user_id=user_id,
                user_message=user_message,
                chat_id=chat_id,
                chat_type=chat_type,
                participant_person_ids=participant_person_ids,
            )

            duration_ms = int((time.monotonic() - start_time) * 1000)
            if memory_context and memory_context.memories:
                logger.info(
                    "memory_retrieval",
                    extra={
                        "memory.count": len(memory_context.memories),
                        "memory.ids": [m.id for m in memory_context.memories],
                        "duration_ms": duration_ms,
                    },
                )
                for mem in memory_context.memories:
                    logger.debug(
                        "  recalled: %s (id=%s, sim=%.2f, meta=%s)",
                        mem.content[:80],
                        mem.id,
                        mem.similarity,
                        mem.metadata,
                    )
            else:
                logger.info(
                    "memory_retrieval",
                    extra={"memory.count": 0, "duration_ms": duration_ms},
                )

            return memory_context

        except Exception:
            logger.warning("memory_retrieval_failed", exc_info=True)
            return None

    async def _list_known_people(
        self,
        user_id: str | None,
    ) -> list[PersonEntry] | None:
        """List known people for the user."""
        if not self._store or not user_id:
            return None

        try:
            return await self._store.list_people(limit=50)
        except Exception:
            logger.warning("known_people_failed", exc_info=True)
            return None

    async def _resolve_sender_person(
        self,
        sender_username: str | None,
        *,
        person_ids: set[str] | None = None,
    ) -> PersonEntry | None:
        """Resolve sender username to canonical person record when possible."""
        if not self._store or not sender_username:
            return None

        try:
            resolved_ids = person_ids
            if resolved_ids is None:
                resolved_ids = await self._resolve_sender_person_ids(sender_username)
            if not resolved_ids:
                return None

            candidates: list[PersonEntry] = []
            for pid in resolved_ids:
                person = await self._store.get_person(pid)
                if person is not None:
                    candidates.append(person)

            if not candidates:
                return None

            # Prefer explicit self-records for this sender; otherwise newest update.
            for person in candidates:
                if any(r.relationship == "self" for r in person.relationships):
                    return person

            candidates.sort(
                key=lambda p: (p.updated_at is not None, p.updated_at, p.id),
                reverse=True,
            )
            return candidates[0]
        except Exception:
            logger.warning("sender_person_resolution_failed", exc_info=True)
            return None

    async def _resolve_sender_person_ids(
        self,
        sender_username: str | None,
    ) -> set[str]:
        if not self._store or not sender_username:
            return set()
        try:
            return await self._store.find_person_ids_for_username(sender_username)
        except Exception:
            logger.warning("sender_person_ids_resolution_failed", exc_info=True)
            return set()
