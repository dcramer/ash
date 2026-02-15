"""Trust model for fact attribution.

Provides a formal model for determining whether a memory is a FACT
(stated by the person it's about) or HEARSAY (stated by someone else
about another person).
"""

from __future__ import annotations

from typing import Literal

# Trust classification type
TrustLevel = Literal["fact", "hearsay"]


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
