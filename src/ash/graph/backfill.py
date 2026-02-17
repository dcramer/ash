"""Edge backfill from legacy FK fields.

Called on first load when edges.jsonl is empty but nodes exist.
Reads FK fields (subject_person_ids, superseded_by_id, person_id,
merged_into) from raw JSON dicts since they've been removed from
the dataclasses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ash.graph.edges import (
    create_about_edge,
    create_has_relationship_edge,
    create_is_person_edge,
    create_merged_into_edge,
    create_stated_by_edge,
    create_supersedes_edge,
)
from ash.graph.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class BackfillResult:
    """Result of edge backfill with tracking of skipped items."""

    created: int = 0
    skipped: list[str] = field(default_factory=list)


def backfill_edges_from_raw(
    graph: KnowledgeGraph,
    raw_memories: list[dict],
    raw_people: list[dict],
    raw_users: list[dict],
) -> BackfillResult:
    """Backfill edges from legacy FK fields in raw JSONL dicts.

    Returns a BackfillResult with created count and details of skipped items.
    """
    result = BackfillResult()

    # Memory → Person (ABOUT) from subject_person_ids
    for d in raw_memories:
        for pid in d.get("subject_person_ids") or []:
            edge = create_about_edge(d["id"], pid, created_by="backfill")
            graph.add_edge(edge)
            result.created += 1

    # Memory → Memory (SUPERSEDES) from superseded_by_id
    for d in raw_memories:
        superseded_by = d.get("superseded_by_id")
        if superseded_by:
            edge = create_supersedes_edge(superseded_by, d["id"], created_by="backfill")
            graph.add_edge(edge)
            result.created += 1

    # User → Person (IS_PERSON) from person_id
    for d in raw_users:
        person_id = d.get("person_id")
        if person_id:
            edge = create_is_person_edge(d["id"], person_id)
            graph.add_edge(edge)
            result.created += 1

    # Person → Person (MERGED_INTO) from merged_into
    merged_ids: set[str] = set()
    for d in raw_people:
        merged_into = d.get("merged_into")
        if merged_into:
            edge = create_merged_into_edge(d["id"], merged_into)
            graph.add_edge(edge)
            merged_ids.add(d["id"])
            result.created += 1

    # Build a username→person_ids lookup from people aliases/names.
    username_to_persons: dict[str, list[str]] = {}
    for person in graph.people.values():
        if person.id in merged_ids:
            continue
        username_to_persons.setdefault(person.name.lower(), []).append(person.id)
        for alias in person.aliases:
            username_to_persons.setdefault(alias.value.lower(), []).append(person.id)

    def _resolve_unique_person(username: str, context: str) -> str | None:
        pids = username_to_persons.get(username.lower())
        if not pids:
            return None
        unique = list(dict.fromkeys(pids))
        if len(unique) == 1:
            return unique[0]
        msg = (
            f"Ambiguous username '{username}' matches {len(unique)} people "
            f"during backfill ({context}), skipping"
        )
        logger.warning(
            "backfill_ambiguous_username",
            extra={
                "username": username,
                "match_count": len(unique),
                "context": context,
            },
        )
        result.skipped.append(msg)
        return None

    # Memory → Person (STATED_BY) from source_username
    for memory in graph.memories.values():
        if memory.source_username:
            pid = _resolve_unique_person(
                memory.source_username, f"STATED_BY for memory {memory.id}"
            )
            if pid:
                edge = create_stated_by_edge(memory.id, pid, created_by="backfill")
                graph.add_edge(edge)
                result.created += 1

    # Person → Person (HAS_RELATIONSHIP) from RelationshipClaim.stated_by
    for person in graph.people.values():
        if person.id in merged_ids:
            continue
        for rc in person.relationships:
            if rc.stated_by and rc.relationship.lower() != "self":
                related_pid = _resolve_unique_person(
                    rc.stated_by,
                    f"HAS_RELATIONSHIP for person {person.id}",
                )
                if related_pid and related_pid != person.id:
                    edge = create_has_relationship_edge(
                        person.id,
                        related_pid,
                        relationship_type=rc.relationship,
                        stated_by=rc.stated_by,
                    )
                    graph.add_edge(edge)
                    result.created += 1

    return result
