"""Trust model for fact attribution.

Provides a formal model for determining whether a memory is a FACT
(stated by the person it's about) or HEARSAY (stated by someone else
about another person).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ash.store.store import Store
    from ash.store.types import MemoryEntry

logger = logging.getLogger(__name__)

# Trust classification type
TrustLevel = Literal["fact", "hearsay"]


@dataclass
class FactSource:
    """Who stated a fact and with what authority.

    Encapsulates speaker identity resolution and trust determination.
    A fact is authoritative (FACT) when the speaker is speaking about
    themselves; otherwise it's HEARSAY.
    """

    person_id: str | None  # Resolved person ID for the speaker
    username: str | None  # Speaker's username
    display_name: str | None  # Speaker's display name
    is_self_statement: bool  # True if fact is about the speaker

    @classmethod
    async def resolve(
        cls,
        user_id: str,
        username: str | None,
        display_name: str | None,
        store: Store,
        subject_person_ids: list[str] | None = None,
    ) -> FactSource:
        """Resolve speaker identity and determine trust level.

        Args:
            user_id: The user ID of the speaker.
            username: The speaker's username (e.g., "notzeeg").
            display_name: The speaker's display name (e.g., "David Cramer").
            store: Store for person ID resolution.
            subject_person_ids: Person IDs the fact is about (empty = about speaker).

        Returns:
            FactSource with resolved speaker identity.
        """
        person_id: str | None = None

        # Resolve speaker's person ID
        if username:
            try:
                person_ids = await store.find_person_ids_for_username(username)
                if person_ids:
                    person_id = next(iter(person_ids))
            except Exception:
                logger.debug(
                    "Failed to resolve speaker person ID",
                    extra={"username": username},
                    exc_info=True,
                )

        # Determine if this is a self-statement
        # A fact with no subjects is implicitly about the speaker
        # A fact where the speaker is in the subjects is also about themselves
        is_self = False
        if not subject_person_ids:
            is_self = True
        elif person_id and person_id in subject_person_ids:
            is_self = True

        return cls(
            person_id=person_id,
            username=username,
            display_name=display_name,
            is_self_statement=is_self,
        )


def classify_trust(
    memory: MemoryEntry,
    source_username: str | None,
) -> TrustLevel:
    """Determine trust level based on source authority.

    A memory is a FACT when the speaker is speaking about themselves
    (no subject_person_ids, or speaker is in subjects). Otherwise
    it's HEARSAY - information about someone stated by another person.

    Args:
        memory: The memory entry to classify.
        source_username: Username of who stated the fact.

    Returns:
        "fact" if authoritative, "hearsay" otherwise.
    """
    # No source attribution = hearsay (we don't know who said it)
    if not source_username:
        return "hearsay"

    # No subjects = speaker talking about themselves = fact
    if not memory.subject_person_ids:
        return "fact"

    # If the memory's source matches its subject, it's a self-statement
    mem_source = (memory.source_username or "").lower()
    if mem_source and mem_source == source_username.lower():
        # Speaker created this memory - but is it about them?
        # We'd need to check if speaker's person_id is in subject_person_ids
        # For now, if the memory has subjects and was created by someone,
        # assume it's hearsay unless we have evidence otherwise
        pass

    # Has subjects = about other people = hearsay
    return "hearsay"


def is_fact_about_self(
    subject_person_ids: list[str] | None,
    speaker_person_id: str | None,
) -> bool:
    """Check if a fact is about the speaker themselves.

    Args:
        subject_person_ids: Person IDs the fact is about.
        speaker_person_id: The speaker's person ID.

    Returns:
        True if the fact is about the speaker (self-statement).
    """
    # No subjects = implicitly about the speaker
    if not subject_person_ids:
        return True

    # Speaker is in subjects = talking about themselves
    if speaker_person_id and speaker_person_id in subject_person_ids:
        return True

    return False
