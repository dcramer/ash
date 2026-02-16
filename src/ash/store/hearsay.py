"""Hearsay management for confirmed facts.

When a user confirms a fact about themselves, any existing hearsay
(statements about them from other people) should be superseded by
the authoritative statement.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ash.store.store import Store
    from ash.store.types import MemoryEntry

logger = logging.getLogger(__name__)


async def supersede_hearsay_for_fact(
    store: Store,
    new_memory: MemoryEntry,
    source_username: str | None,
) -> int:
    """Convenience function to supersede hearsay for a confirmed fact.

    This is the main entry point for hearsay supersession during
    memory extraction. It resolves the speaker's person IDs and
    delegates to the store's supersession logic.

    Args:
        store: Store for memory and person operations.
        new_memory: The newly created memory.
        source_username: Username of who stated the fact.

    Returns:
        Number of hearsay memories superseded.
    """
    # Caller already verified this is a self-fact. The memory may have
    # the speaker's person_id injected into subject_person_ids for graph
    # traversal, so we don't re-check that field here.
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
