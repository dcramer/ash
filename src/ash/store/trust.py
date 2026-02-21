"""Trust model for fact attribution using graph edges.

Classifies memories as FACT or HEARSAY based on STATED_BY and ABOUT edges:
- FACT: The memory was STATED_BY a person who is also a subject (ABOUT edge)
- HEARSAY: The memory was stated by someone other than the subject
- UNKNOWN: No STATED_BY edge exists (missing attribution or agent-provided)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ash.graph.graph import KnowledgeGraph

# Trust classification type
TrustLevel = Literal["fact", "hearsay", "unknown"]


def classify_trust(graph: KnowledgeGraph, memory_id: str) -> TrustLevel:
    """Classify a memory's trust level using graph edges.

    Uses STATED_BY and ABOUT edges to determine if the speaker
    is also a subject of the memory (first-person = fact).

    Args:
        graph: The knowledge graph.
        memory_id: Memory ID to classify.

    Returns:
        "fact" if stated by a subject, "hearsay" if stated by non-subject,
        "unknown" if no STATED_BY edge exists.
    """
    from ash.graph.edges import get_stated_by_person, get_subject_person_ids

    stated_by_pid = get_stated_by_person(graph, memory_id)
    if stated_by_pid is None:
        return "unknown"

    subject_pids = get_subject_person_ids(graph, memory_id)

    # No subjects = implicitly about the speaker (self-fact)
    if not subject_pids:
        return "fact"

    # Speaker is one of the subjects = first-person statement
    if stated_by_pid in subject_pids:
        return "fact"

    return "hearsay"


def is_fact_about_self(
    subject_person_ids: list[str] | None,
    speaker_person_id: str | None,
) -> bool:
    """Check if a fact is about the speaker themselves.

    Simple heuristic for use during extraction (before edges exist).

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


def get_trust_weight(trust_level: TrustLevel) -> float:
    """Get a retrieval weight multiplier for a trust level.

    Higher trust = higher weight in retrieval scoring.

    Args:
        trust_level: The trust classification.

    Returns:
        Weight multiplier (0.0 to 1.0).
    """
    if trust_level == "fact":
        return 1.0
    if trust_level == "hearsay":
        return 0.8
    return 0.9  # unknown - slightly below fact
