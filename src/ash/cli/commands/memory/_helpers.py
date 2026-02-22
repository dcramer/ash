"""Shared helpers for memory CLI commands."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.store.store import Store
    from ash.store.types import PersonEntry

logger = logging.getLogger(__name__)


def _matches_username(person: PersonEntry, username: str) -> bool:
    """Check if a person matches a username (case-insensitive)."""
    return person.matches_username(username)


def is_source_self_reference(
    source_username: str | None,
    owner_user_id: str | None,
    subject_person_ids: list[str] | None,
    all_people: list[PersonEntry],
    people_by_id: dict[str, PersonEntry],
) -> bool:
    """Determine if the source user is speaking about themselves.

    Returns True if:
    - source matches a self-person (person with relationship "self")
    - source matches any subject person (source is the subject)
    """
    if not source_username:
        return False

    source_id = source_username.lower()

    # Check if source matches a self-person (globally, not per-owner)
    for person in all_people:
        if any(rc.relationship == "self" for rc in person.relationships):
            if _matches_username(person, source_id):
                return True

    # Check if source matches any subject person
    if subject_person_ids:
        for person_id in subject_person_ids:
            person = people_by_id.get(person_id)
            if person and _matches_username(person, source_id):
                return True

    return False


async def get_store(config: AshConfig) -> Store | None:
    """Create a Store from CLI context.

    Returns None if embeddings are not configured.
    """
    from ash.cli.context import get_graph_dir
    from ash.memory.runtime import initialize_memory_runtime

    graph_dir = get_graph_dir()
    runtime = await initialize_memory_runtime(
        config=config,
        graph_dir=graph_dir,
        model_alias="default",
        initialize_extractor=False,
        logger=logger,
    )
    return runtime.store
