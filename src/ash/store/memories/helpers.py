"""Shared helpers for memory operations."""

from __future__ import annotations

from ash.graph.edges import ABOUT


def load_subjects_from_graph(graph, memory_id: str) -> list[str]:
    """Load subject_person_ids for a memory from ABOUT edges."""
    edges = graph.get_outgoing(memory_id, edge_type=ABOUT)
    return [e.target_id for e in edges]


def load_subjects_batch_from_graph(
    graph, memory_ids: list[str]
) -> dict[str, list[str]]:
    """Load subject_person_ids for multiple memories from ABOUT edges."""
    result: dict[str, list[str]] = {}
    for mid in memory_ids:
        subjects = load_subjects_from_graph(graph, mid)
        if subjects:
            result[mid] = subjects
    return result
