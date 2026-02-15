"""Shared helpers for people operations."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text

from ash.store.mappers import row_to_person as _row_to_person
from ash.store.types import (
    AliasEntry,
    PersonEntry,
    RelationshipClaim,
    _parse_datetime,
)

# Sentinel for sort key when created_at is None
EPOCH = datetime.min.replace(tzinfo=UTC)

# SQL clause to normalize alias values the same way as normalize_reference:
# lowercase, trim, then strip 'my ', 'the ', '@' prefixes.
ALIAS_NORM_MATCH = """(LOWER(TRIM(pa.value)) = :ref
    OR (LOWER(TRIM(pa.value)) LIKE 'my %' AND SUBSTR(LOWER(TRIM(pa.value)), 4) = :ref)
    OR (LOWER(TRIM(pa.value)) LIKE 'the %' AND SUBSTR(LOWER(TRIM(pa.value)), 5) = :ref)
    OR (LOWER(TRIM(pa.value)) LIKE '@%' AND SUBSTR(LOWER(TRIM(pa.value)), 2) = :ref))"""

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


async def load_person_full(session, person_id: str) -> PersonEntry | None:
    """Load a person with aliases and relationships."""
    result = await session.execute(
        text("SELECT * FROM people WHERE id = :id"),
        {"id": person_id},
    )
    row = result.fetchone()
    if not row:
        return None

    aliases = await load_aliases(session, person_id)
    relationships = await load_relationships(session, person_id)
    return _row_to_person(row, aliases, relationships)


async def load_aliases(session, person_id: str) -> list[AliasEntry]:
    result = await session.execute(
        text(
            "SELECT value, added_by, created_at FROM person_aliases WHERE person_id = :id"
        ),
        {"id": person_id},
    )
    return [
        AliasEntry(
            value=row[0],
            added_by=row[1],
            created_at=_parse_datetime(row[2]),
        )
        for row in result.fetchall()
    ]


async def load_relationships(session, person_id: str) -> list[RelationshipClaim]:
    result = await session.execute(
        text(
            "SELECT relationship, stated_by, created_at FROM person_relationships WHERE person_id = :id"
        ),
        {"id": person_id},
    )
    return [
        RelationshipClaim(
            relationship=row[0],
            stated_by=row[1],
            created_at=_parse_datetime(row[2]),
        )
        for row in result.fetchall()
    ]


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
