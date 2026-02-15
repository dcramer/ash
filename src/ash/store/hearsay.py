"""Hearsay management for confirmed facts.

When a user confirms a fact about themselves, any existing hearsay
(statements about them from other people) should be superseded by
the authoritative statement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ash.store.trust import is_fact_about_self

if TYPE_CHECKING:
    from ash.store.store import Store
    from ash.store.types import MemoryEntry

logger = logging.getLogger(__name__)


@dataclass
class HearsayResult:
    """Result of hearsay supersession operation."""

    superseded_count: int  # Number of hearsay memories superseded
    checked_count: int  # Number of hearsay memories checked


class HearsayManager:
    """Manages supersession of hearsay by confirmed facts.

    When a user speaks about themselves (a FACT), this manager finds
    and supersedes any existing HEARSAY about them - statements made
    by other people that the user is now confirming or correcting.
    """

    def __init__(self, store: Store):
        """Initialize hearsay manager.

        Args:
            store: Store for memory operations.
        """
        self._store = store

    async def handle_confirmed_fact(
        self,
        new_memory: MemoryEntry,
        speaker_person_ids: set[str],
        source_username: str,
        owner_user_id: str,
    ) -> HearsayResult:
        """Supersede hearsay when user confirms fact about themselves.

        When a user speaks about themselves (no subject_person_ids), this
        finds hearsay memories about them (same person, different source)
        and supersedes them with the authoritative statement.

        Args:
            new_memory: The newly created memory (the confirmed fact).
            speaker_person_ids: Person IDs associated with the speaker.
            source_username: Username of who stated the fact.
            owner_user_id: Owner user ID for scoping.

        Returns:
            HearsayResult with counts of checked and superseded memories.
        """
        # Only handle self-facts (speaker talking about themselves)
        if not is_fact_about_self(new_memory.subject_person_ids, None):
            return HearsayResult(superseded_count=0, checked_count=0)

        if not speaker_person_ids:
            return HearsayResult(superseded_count=0, checked_count=0)

        try:
            superseded = await self._store.supersede_confirmed_hearsay(
                new_memory=new_memory,
                person_ids=speaker_person_ids,
                source_username=source_username,
                owner_user_id=owner_user_id,
            )
            return HearsayResult(
                superseded_count=superseded,
                checked_count=0,  # We don't track checked count currently
            )
        except Exception:
            logger.warning(
                "Failed to check for hearsay supersession",
                exc_info=True,
            )
            return HearsayResult(superseded_count=0, checked_count=0)


async def supersede_hearsay_for_fact(
    store: Store,
    new_memory: MemoryEntry,
    source_username: str | None,
    owner_user_id: str,
) -> int:
    """Convenience function to supersede hearsay for a confirmed fact.

    This is the main entry point for hearsay supersession during
    memory extraction. It resolves the speaker's person IDs and
    delegates to the store's supersession logic.

    Args:
        store: Store for memory and person operations.
        new_memory: The newly created memory.
        source_username: Username of who stated the fact.
        owner_user_id: Owner user ID for scoping.

    Returns:
        Number of hearsay memories superseded.
    """
    # Only check for hearsay when this is a FACT (user speaking about themselves)
    # A fact has no subject_person_ids (implicitly about the speaker)
    if new_memory.subject_person_ids:
        return 0

    if not source_username:
        return 0

    try:
        # Resolve the source user's person IDs for hearsay lookup
        source_person_ids = await store.find_person_ids_for_username(source_username)
        if not source_person_ids:
            return 0

        superseded = await store.supersede_confirmed_hearsay(
            new_memory=new_memory,
            person_ids=source_person_ids,
            source_username=source_username,
            owner_user_id=owner_user_id,
        )

        if superseded > 0:
            logger.debug(
                "Superseded %d hearsay memories with confirmed fact",
                superseded,
            )

        return superseded

    except Exception:
        logger.warning(
            "Failed to check for hearsay supersession",
            exc_info=True,
        )
        return 0
