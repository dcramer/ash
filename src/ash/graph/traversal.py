"""Graph traversal algorithms for multi-hop retrieval."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from ash.graph.graph import Edge, KnowledgeGraph

# Edge types to skip by default during traversal
DEFAULT_EXCLUDE_EDGE_TYPES = {"SUPERSEDES"}


@dataclass
class TraversalResult:
    """Result of a graph traversal."""

    node_id: str
    node_type: str
    hops: int  # Distance from seed
    path: list[str] = field(default_factory=list)  # Edge IDs traversed


def bfs_traverse(
    graph: KnowledgeGraph,
    seed_ids: set[str],
    max_hops: int = 2,
    exclude_edge_types: set[str] | None = None,
    filter_fn: Callable[[str, Edge], bool] | None = None,
) -> list[TraversalResult]:
    """BFS through adjacency lists from seed nodes.

    Follows both outgoing and incoming edges.
    Excludes SUPERSEDES by default.
    filter_fn is called per-hop for privacy/scope filtering.
    """
    if exclude_edge_types is None:
        exclude_edge_types = DEFAULT_EXCLUDE_EDGE_TYPES

    visited: set[str] = set(seed_ids)
    results: list[TraversalResult] = []

    # Queue entries: (node_id, hops, path_of_edge_ids)
    queue: deque[tuple[str, int, list[str]]] = deque()

    for seed_id in seed_ids:
        queue.append((seed_id, 0, []))

    while queue:
        node_id, hops, path = queue.popleft()

        if hops > 0:
            node_type = graph._node_type.get(node_id, "unknown")
            results.append(
                TraversalResult(
                    node_id=node_id,
                    node_type=node_type,
                    hops=hops,
                    path=path,
                )
            )

        if hops >= max_hops:
            continue

        # Follow outgoing edges
        for edge in graph.get_outgoing(node_id, active_only=True):
            if edge.edge_type in exclude_edge_types:
                continue
            if filter_fn and not filter_fn(edge.target_id, edge):
                continue
            if edge.target_id not in visited:
                visited.add(edge.target_id)
                queue.append((edge.target_id, hops + 1, [*path, edge.id]))

        # Follow incoming edges
        for edge in graph.get_incoming(node_id, active_only=True):
            if edge.edge_type in exclude_edge_types:
                continue
            if filter_fn and not filter_fn(edge.source_id, edge):
                continue
            if edge.source_id not in visited:
                visited.add(edge.source_id)
                queue.append((edge.source_id, hops + 1, [*path, edge.id]))

    return results
