"""Shared helpers for people operations."""

from __future__ import annotations

from datetime import UTC, datetime

from ash.store.types import PersonEntry

# Sentinel for sort key when created_at is None
EPOCH = datetime.min.replace(tzinfo=UTC)

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

FUZZY_MATCH_PROMPT = """Given a person reference and a list of known people, determine if the reference matches any existing person.

Reference: "{reference}"
{context_section}
{speaker_section}
Known people:
{people_list}

Consider: name variants (first name <-> full name, nicknames), relationship links (e.g., "Sarah" from speaker "dcramer" matches a person with relationship "wife" stated by "dcramer"), and alias matches. Prefer matching relationships stated by the current speaker.

If the reference clearly refers to one of the known people, respond with ONLY the ID.
If no clear match, respond with NONE.

Respond with only the ID or NONE, nothing else."""


def normalize_reference(text_value: str) -> str:
    """Normalize a person reference for matching."""
    result = text_value.lower().strip()
    for prefix in ("my ", "the ", "@"):
        result = result.removeprefix(prefix)
    return result


def primary_sort_key(p: PersonEntry) -> tuple[int, datetime]:
    """Sort key for picking primary person in merge/dedup."""
    score = len(p.aliases) + len(p.relationships)
    return (-score, p.created_at or EPOCH)
