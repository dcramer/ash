"""Shared helpers for memory CLI commands."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.db.engine import Database
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


async def get_store(config: AshConfig, db: Database) -> Store | None:
    """Create a Store from CLI context.

    Returns None if embeddings are not configured.
    """
    from ash.llm.registry import create_registry
    from ash.store import create_store

    if not config.embeddings:
        return None

    embeddings_key = config.resolve_embeddings_api_key()
    if not embeddings_key:
        return None

    # Build registry with embedding provider key (and Anthropic for LLM verification)
    default_model = config.get_model("default")
    if default_model.provider == "anthropic":
        anthropic_key = config.resolve_api_key("default")
    else:
        anthropic_key = config._resolve_provider_api_key("anthropic")

    llm_registry = create_registry(
        anthropic_api_key=anthropic_key.get_secret_value() if anthropic_key else None,
        openai_api_key=embeddings_key.get_secret_value()
        if config.embeddings.provider == "openai"
        else None,
    )

    return await create_store(
        db=db,
        llm_registry=llm_registry,
        embedding_model=config.embeddings.model,
        embedding_provider=config.embeddings.provider,
        max_entries=config.memory.max_entries,
    )
