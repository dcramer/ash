"""Edge type constants and helper functions for creating typed edges.

All graph relationships are represented as edges. This module provides:
- Constants for edge type names
- Factory functions for creating edges with proper type annotations
- Query helpers for common edge-based lookups
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ash.graph.graph import Edge

if TYPE_CHECKING:
    from ash.graph.graph import KnowledgeGraph

# Edge type constants
ABOUT = "ABOUT"  # Memory → Person (memory is about this person)
STATED_BY = "STATED_BY"  # Memory → Person (person stated this fact)
SUPERSEDES = "SUPERSEDES"  # Memory → Memory (new supersedes old)
IS_PERSON = "IS_PERSON"  # User → Person (user identity maps to person)
MERGED_INTO = "MERGED_INTO"  # Person → Person (source merged into target)
HAS_RELATIONSHIP = "HAS_RELATIONSHIP"  # Person → Person (relationship link)
LEARNED_IN = "LEARNED_IN"  # Memory → Chat (where a memory was first learned)
PARTICIPATES_IN = "PARTICIPATES_IN"  # Person → Chat (chat membership)


def _make_edge_id() -> str:
    return f"e-{uuid.uuid4().hex}"


def create_about_edge(
    memory_id: str,
    person_id: str,
    *,
    created_by: str | None = None,
) -> Edge:
    """Create an ABOUT edge: Memory → Person."""
    return Edge(
        id=_make_edge_id(),
        edge_type=ABOUT,
        source_type="memory",
        source_id=memory_id,
        target_type="person",
        target_id=person_id,
        created_at=datetime.now(UTC),
        created_by=created_by,
    )


def create_supersedes_edge(
    new_memory_id: str,
    old_memory_id: str,
    *,
    created_by: str | None = None,
) -> Edge:
    """Create a SUPERSEDES edge: NewMemory → OldMemory."""
    return Edge(
        id=_make_edge_id(),
        edge_type=SUPERSEDES,
        source_type="memory",
        source_id=new_memory_id,
        target_type="memory",
        target_id=old_memory_id,
        created_at=datetime.now(UTC),
        created_by=created_by,
    )


def create_is_person_edge(
    user_id: str,
    person_id: str,
) -> Edge:
    """Create an IS_PERSON edge: User → Person."""
    return Edge(
        id=_make_edge_id(),
        edge_type=IS_PERSON,
        source_type="user",
        source_id=user_id,
        target_type="person",
        target_id=person_id,
        created_at=datetime.now(UTC),
    )


def create_stated_by_edge(
    memory_id: str,
    person_id: str,
    *,
    created_by: str | None = None,
) -> Edge:
    """Create a STATED_BY edge: Memory → Person."""
    return Edge(
        id=_make_edge_id(),
        edge_type=STATED_BY,
        source_type="memory",
        source_id=memory_id,
        target_type="person",
        target_id=person_id,
        created_at=datetime.now(UTC),
        created_by=created_by,
    )


def create_merged_into_edge(
    source_person_id: str,
    target_person_id: str,
) -> Edge:
    """Create a MERGED_INTO edge: Person → Person."""
    return Edge(
        id=_make_edge_id(),
        edge_type=MERGED_INTO,
        source_type="person",
        source_id=source_person_id,
        target_type="person",
        target_id=target_person_id,
        created_at=datetime.now(UTC),
    )


def create_has_relationship_edge(
    person_id: str,
    related_person_id: str,
    *,
    relationship_type: str | None = None,
    stated_by: str | None = None,
) -> Edge:
    """Create a HAS_RELATIONSHIP edge: Person → Person."""
    properties: dict[str, str] = {}
    if relationship_type:
        properties["relationship_type"] = relationship_type
    if stated_by:
        properties["stated_by"] = stated_by
    return Edge(
        id=_make_edge_id(),
        edge_type=HAS_RELATIONSHIP,
        source_type="person",
        source_id=person_id,
        target_type="person",
        target_id=related_person_id,
        properties=properties or None,
        created_at=datetime.now(UTC),
    )


# -- Query helpers --


def get_subject_person_ids(graph: KnowledgeGraph, memory_id: str) -> list[str]:
    """Get person IDs that a memory is about (via ABOUT edges)."""
    edges = graph.get_outgoing(memory_id, edge_type=ABOUT)
    return [e.target_id for e in edges]


def get_subject_person_ids_batch(
    graph: KnowledgeGraph, memory_ids: list[str]
) -> dict[str, list[str]]:
    """Get subject person IDs for multiple memories."""
    result: dict[str, list[str]] = {}
    for mid in memory_ids:
        subjects = get_subject_person_ids(graph, mid)
        if subjects:
            result[mid] = subjects
    return result


def get_memories_about_person(graph: KnowledgeGraph, person_id: str) -> list[str]:
    """Get memory IDs that are about a person (via incoming ABOUT edges)."""
    edges = graph.get_incoming(person_id, edge_type=ABOUT)
    return [e.source_id for e in edges]


def get_superseded_by(graph: KnowledgeGraph, memory_id: str) -> str | None:
    """Get the memory that supersedes this one (via incoming SUPERSEDES edge)."""
    edges = graph.get_incoming(memory_id, edge_type=SUPERSEDES)
    return edges[0].source_id if edges else None


def get_supersession_targets(graph: KnowledgeGraph, memory_id: str) -> list[str]:
    """Get memories that this memory supersedes (via outgoing SUPERSEDES edges)."""
    edges = graph.get_outgoing(memory_id, edge_type=SUPERSEDES)
    return [e.target_id for e in edges]


def get_person_for_user(graph: KnowledgeGraph, user_id: str) -> str | None:
    """Get person ID linked to a user (via IS_PERSON edge)."""
    edges = graph.get_outgoing(user_id, edge_type=IS_PERSON)
    return edges[0].target_id if edges else None


def get_users_for_person(graph: KnowledgeGraph, person_id: str) -> list[str]:
    """Get user IDs linked to a person (via incoming IS_PERSON edges)."""
    edges = graph.get_incoming(person_id, edge_type=IS_PERSON)
    return [e.source_id for e in edges]


def get_merged_into(graph: KnowledgeGraph, person_id: str) -> str | None:
    """Get person this was merged into (via MERGED_INTO edge)."""
    edges = graph.get_outgoing(person_id, edge_type=MERGED_INTO)
    return edges[0].target_id if edges else None


def follow_merge_chain(
    graph: KnowledgeGraph, person_id: str, max_depth: int = 10
) -> str:
    """Follow MERGED_INTO edges to find the final (canonical) person ID.

    Uses a visited set to detect cycles and a max_depth guard.
    """
    visited: set[str] = set()
    current = person_id
    for _ in range(max_depth):
        visited.add(current)
        merged = get_merged_into(graph, current)
        if not merged or merged in visited:
            return current
        current = merged
    return current


def get_stated_by_person(graph: KnowledgeGraph, memory_id: str) -> str | None:
    """Get the person who stated a memory (via STATED_BY edge)."""
    edges = graph.get_outgoing(memory_id, edge_type=STATED_BY)
    return edges[0].target_id if edges else None


def get_memories_stated_by(graph: KnowledgeGraph, person_id: str) -> list[str]:
    """Get memory IDs stated by a person (via incoming STATED_BY edges)."""
    edges = graph.get_incoming(person_id, edge_type=STATED_BY)
    return [e.source_id for e in edges]


def get_relationships_for_person(graph: KnowledgeGraph, person_id: str) -> list[Edge]:
    """Get HAS_RELATIONSHIP edges for a person (both directions)."""
    outgoing = graph.get_outgoing(person_id, edge_type=HAS_RELATIONSHIP)
    incoming = graph.get_incoming(person_id, edge_type=HAS_RELATIONSHIP)
    return outgoing + incoming


def get_related_people(graph: KnowledgeGraph, person_id: str) -> list[str]:
    """Get person IDs related to a person (via HAS_RELATIONSHIP edges)."""
    edges = get_relationships_for_person(graph, person_id)
    result: list[str] = []
    for e in edges:
        other = e.target_id if e.source_id == person_id else e.source_id
        if other not in result:
            result.append(other)
    return result


# -- LEARNED_IN helpers --


def create_learned_in_edge(
    memory_id: str,
    chat_id: str,
    *,
    created_by: str | None = None,
) -> Edge:
    """Create a LEARNED_IN edge: Memory → Chat."""
    return Edge(
        id=_make_edge_id(),
        edge_type=LEARNED_IN,
        source_type="memory",
        source_id=memory_id,
        target_type="chat",
        target_id=chat_id,
        created_at=datetime.now(UTC),
        created_by=created_by,
    )


def get_learned_in_chat(graph: KnowledgeGraph, memory_id: str) -> str | None:
    """Get the chat ID where this memory was learned."""
    edges = graph.get_outgoing(memory_id, edge_type=LEARNED_IN)
    return edges[0].target_id if edges else None


# -- PARTICIPATES_IN helpers --


def create_participates_in_edge(
    person_id: str,
    chat_id: str,
) -> Edge:
    """Create a PARTICIPATES_IN edge: Person → Chat."""
    return Edge(
        id=_make_edge_id(),
        edge_type=PARTICIPATES_IN,
        source_type="person",
        source_id=person_id,
        target_type="chat",
        target_id=chat_id,
        created_at=datetime.now(UTC),
    )


def get_chat_participant_person_ids(graph: KnowledgeGraph, chat_id: str) -> set[str]:
    """Get all person IDs that participate in a chat."""
    edges = graph.get_incoming(chat_id, edge_type=PARTICIPATES_IN)
    return {e.source_id for e in edges}


def person_participates_in_chat(
    graph: KnowledgeGraph, person_id: str, chat_id: str
) -> bool:
    """Check if a person participates in a specific chat."""
    edges = graph.get_outgoing(person_id, edge_type=PARTICIPATES_IN)
    return any(e.target_id == chat_id for e in edges)
