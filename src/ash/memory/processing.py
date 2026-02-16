"""Shared fact-processing logic for memory extraction.

Both the active extraction path (agent.py) and passive extraction path
(passive.py) share the same post-extraction processing steps:
subject resolution, self-person injection, hearsay supersession,
sensitivity/portable passthrough, shared vs personal ownership,
relationship extraction, owner filtering, speaker validation,
post-extraction dedup, and existing memory dedup.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ash.core.filters import build_owner_matchers, is_owner_name
from ash.store.types import MemoryType

if TYPE_CHECKING:
    from ash.store.store import Store
    from ash.store.types import ExtractedFact, PersonEntry

logger = logging.getLogger(__name__)

# Invalid speaker values that indicate assistant attribution
_INVALID_SPEAKERS = frozenset({"agent", "assistant", "bot", "system", "ash"})


def validate_speaker(speaker: str | None) -> str | None:
    """Validate speaker, filtering out invalid values.

    Returns None for invalid speakers (agent, assistant, etc.) or empty values.
    Preserves original casing -- callers already lowercase for comparison.
    """
    if not speaker:
        return None
    if speaker.lower() in _INVALID_SPEAKERS:
        logger.debug("Filtering invalid speaker: %s", speaker)
        return None
    return speaker


def extract_relationship_term(content: str) -> str | None:
    """Extract a relationship term from fact content.

    Scans for known relationship terms (wife, boss, friend, etc.) to
    attach to person records when a RELATIONSHIP-type fact is extracted.
    Returns the first match found, or None.
    """
    from ash.store.people import RELATIONSHIP_TERMS

    content_lower = content.lower()
    # Check multi-word terms first (e.g., "best friend" before "friend")
    for term in sorted(RELATIONSHIP_TERMS, key=lambda t: len(t), reverse=True):
        if term in content_lower:
            return term
    return None


async def ensure_self_person(
    store: Store,
    user_id: str,
    username: str,
    display_name: str,
) -> str | None:
    """Ensure a self-Person exists for the user with username as alias.

    This enables proper trust determination by linking the username
    (used as source_username) to the display name (used for display).

    Lookup order: display_name first, then username. If a matching person
    exists but lacks a "self" relationship, we claim it (this handles
    the case where another user mentions "David Cramer" before David
    speaks). If no match, create a new self-person.

    Args:
        store: Store with people operations.
        user_id: The user ID (used as created_by for new records).
        username: The user's handle/username (e.g., "notzeeg").
        display_name: The user's display name (e.g., "David Cramer").

    Returns:
        The person_id for the self-person, or None on failure.
    """
    async with store._self_person_lock:
        try:
            # Try display name first, then username
            existing = await store.find_person(display_name)
            if not existing and username:
                existing = await store.find_person(username)

            if existing:
                is_self = any(
                    rc.relationship == "self" for rc in existing.relationships
                )
                if not is_self:
                    await store.add_relationship(
                        existing.id, "self", stated_by=username
                    )
                await _sync_person_details(
                    store, existing, display_name, username, user_id
                )
                return existing.id

            # No match found -- create new self-person
            # When no username, use numeric user_id as alias to reconnect the graph
            aliases = [username] if username else [user_id]
            new_person = await store.create_person(
                created_by=user_id,
                name=display_name,
                relationship="self",
                aliases=aliases,
                relationship_stated_by=username or None,
            )
            logger.debug(
                "Created self-person for user",
                extra={
                    "user_id": user_id,
                    "person_name": display_name,
                    "username": username,
                },
            )

            # Dedup: merge new self-person against any existing person with same name.
            # Use exclude_self=False because the new person always has "self" relationship
            # and would be skipped otherwise.
            result_id = new_person.id
            try:
                candidates = await store.find_dedup_candidates(
                    [new_person.id], exclude_self=False
                )
                for primary_id, secondary_id in candidates:
                    await store.merge_people(primary_id, secondary_id)
                    if secondary_id == new_person.id:
                        result_id = primary_id
            except Exception:
                logger.warning("Self-person dedup failed", exc_info=True)
            return result_id
        except Exception:
            logger.warning("Failed to ensure self-person", exc_info=True)
            return None


async def _sync_person_details(
    store: Store,
    person: PersonEntry,
    display_name: str,
    username: str,
    user_id: str,
) -> None:
    """Update a person's name and ensure username alias exists."""
    if display_name and person.name != display_name:
        await store.update_person(
            person_id=person.id, name=display_name, updated_by=user_id
        )
    if username:
        aliases_lower = [a.value.lower() for a in person.aliases]
        if username.lower() not in aliases_lower:
            await store.add_alias(person.id, username, user_id)


async def enrich_owner_names(
    store: Store,
    owner_names: list[str],
    speaker_person_id: str,
) -> None:
    """Add person aliases to owner_names for better owner filtering.

    Mutates owner_names in place.
    """
    person = await store.get_person(speaker_person_id)
    if person:
        existing = {n.lower() for n in owner_names}
        for alias in person.aliases:
            if alias.value.lower() not in existing:
                owner_names.append(alias.value)


async def process_extracted_facts(
    facts: list[ExtractedFact],
    store: Store,
    user_id: str,
    chat_id: str | None = None,
    speaker_username: str | None = None,
    speaker_display_name: str | None = None,
    speaker_person_id: str | None = None,
    owner_names: list[str] | None = None,
    source: str = "background_extraction",
    confidence_threshold: float = 0.7,
) -> list[str]:
    """Process extracted facts through the full post-extraction pipeline.

    Handles: subject resolution, self-person injection, hearsay supersession,
    sensitivity/portable passthrough, shared vs personal ownership,
    relationship extraction, owner filtering, speaker validation,
    post-extraction dedup, and existing memory dedup.

    Returns:
        List of stored memory IDs.
    """
    owner_matchers = build_owner_matchers(owner_names)
    newly_created_person_ids: list[str] = []
    stored_ids: list[str] = []

    for fact in facts:
        if fact.confidence < confidence_threshold:
            continue

        try:
            subject_person_ids: list[str] | None = None
            subject_to_pid: dict[str, str] = {}
            if fact.subjects:
                subject_person_ids = []
                for subject in fact.subjects:
                    if is_owner_name(subject, owner_matchers):
                        logger.debug("Skipping owner as subject: %s", subject)
                        continue
                    try:
                        result = await store.resolve_or_create_person(
                            created_by=user_id,
                            reference=subject,
                            content_hint=fact.content,
                            relationship_stated_by=speaker_username,
                        )
                        subject_person_ids.append(result.person_id)
                        subject_to_pid[subject.lower()] = result.person_id
                        if result.created:
                            newly_created_person_ids.append(result.person_id)
                    except Exception:
                        logger.warning("Failed to resolve subject: %s", subject)

            # For RELATIONSHIP facts, attach the term to the person record
            if fact.memory_type == MemoryType.RELATIONSHIP and subject_person_ids:
                rel_term = extract_relationship_term(fact.content)
                if rel_term:
                    for pid in subject_person_ids:
                        try:
                            await store.add_relationship(
                                pid,
                                rel_term,
                                stated_by=speaker_username,
                            )
                        except Exception:
                            logger.debug(
                                "Failed to add relationship %s to %s",
                                rel_term,
                                pid,
                            )

            # Register explicit aliases from extraction
            if fact.aliases and subject_person_ids:
                for alias_subject, alias_values in fact.aliases.items():
                    pid = subject_to_pid.get(alias_subject.lower())
                    if not pid:
                        continue
                    for alias_val in alias_values:
                        try:
                            await store.add_alias(
                                pid,
                                alias_val,
                                added_by=speaker_username or user_id,
                            )
                        except Exception:
                            logger.debug(
                                "Failed to add alias %s to person %s",
                                alias_val,
                                pid,
                            )

            # Capture whether this is a self-fact before injecting speaker_person_id
            is_self_fact = not subject_person_ids

            # Self-facts should reference the speaker's person record
            # for graph traversal. Skip RELATIONSHIP type.
            if (
                is_self_fact
                and speaker_person_id
                and fact.memory_type != MemoryType.RELATIONSHIP
            ):
                subject_person_ids = [speaker_person_id]

            # Filter out invalid speaker values
            speaker = validate_speaker(fact.speaker)

            # Determine source user from extracted speaker or session
            source_username = speaker or speaker_username or user_id
            source_display_name = (
                speaker_display_name if source_username == speaker_username else None
            )

            new_memory = await store.add_memory(
                content=fact.content,
                source=source,
                memory_type=fact.memory_type,
                owner_user_id=user_id if not fact.shared else None,
                chat_id=chat_id if fact.shared else None,
                subject_person_ids=subject_person_ids or None,
                observed_at=datetime.now(UTC),
                source_username=source_username,
                source_display_name=source_display_name,
                extraction_confidence=fact.confidence,
                sensitivity=fact.sensitivity,
                portable=fact.portable,
            )

            logger.debug(
                "Extracted memory: %s (confidence=%.2f, speaker=%s)",
                fact.content[:50],
                fact.confidence,
                source_username,
            )
            stored_ids.append(new_memory.id)

            # Check for hearsay to supersede when this is a self-fact
            if is_self_fact and source_username:
                from ash.store.hearsay import supersede_hearsay_for_fact

                await supersede_hearsay_for_fact(
                    store=store,
                    new_memory=new_memory,
                    source_username=source_username,
                )
        except Exception:
            logger.debug(
                "Failed to store extracted fact: %s",
                fact.content[:50],
                exc_info=True,
            )

    # Post-extraction dedup: merge newly created people that match existing
    if newly_created_person_ids:
        try:
            candidates = await store.find_dedup_candidates(
                newly_created_person_ids, exclude_self=True
            )
            for primary_id, secondary_id in candidates:
                await store.merge_people(primary_id, secondary_id)
        except Exception:
            logger.warning("Post-extraction dedup failed", exc_info=True)

    return stored_ids
