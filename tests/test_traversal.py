"""Tests for BFS graph traversal."""

from __future__ import annotations

from ash.graph.edges import (
    ABOUT,
    create_about_edge,
    create_merged_into_edge,
    create_supersedes_edge,
)
from ash.graph.graph import KnowledgeGraph
from ash.graph.traversal import bfs_traverse
from ash.store.types import MemoryEntry, MemoryType, PersonEntry


def _build_graph() -> KnowledgeGraph:
    """Build a test graph:

    person-alice ←ABOUT— mem-alice-cats ("Alice likes cats")
    person-alice ←ABOUT— mem-alice-work ("Alice works at Acme")
    person-bob   ←ABOUT— mem-bob-music ("Bob likes music")
    person-bob   ←ABOUT— mem-bob-alice ("Bob is Alice's friend")
    person-alice ←ABOUT— mem-bob-alice ("Bob is Alice's friend")
    mem-old —SUPERSEDES→ mem-new (supersession chain)
    person-dup —MERGED_INTO→ person-alice
    """
    graph = KnowledgeGraph()

    # People
    for pid in ["person-alice", "person-bob", "person-dup"]:
        graph.add_person(PersonEntry(id=pid, name=pid.split("-")[1].title()))

    # Memories
    memories = {
        "mem-alice-cats": "Alice likes cats",
        "mem-alice-work": "Alice works at Acme",
        "mem-bob-music": "Bob likes music",
        "mem-bob-alice": "Bob is Alice's friend",
        "mem-old": "Old version",
        "mem-new": "New version",
    }
    for mid, content in memories.items():
        graph.add_memory(
            MemoryEntry(id=mid, content=content, memory_type=MemoryType.KNOWLEDGE)
        )

    # ABOUT edges
    graph.add_edge(create_about_edge("mem-alice-cats", "person-alice"))
    graph.add_edge(create_about_edge("mem-alice-work", "person-alice"))
    graph.add_edge(create_about_edge("mem-bob-music", "person-bob"))
    graph.add_edge(create_about_edge("mem-bob-alice", "person-bob"))
    graph.add_edge(create_about_edge("mem-bob-alice", "person-alice"))

    # SUPERSEDES edge
    graph.add_edge(create_supersedes_edge("mem-new", "mem-old"))

    # MERGED_INTO edge
    graph.add_edge(create_merged_into_edge("person-dup", "person-alice"))

    return graph


class TestBFSTraversal:
    def test_single_hop_from_person(self):
        """BFS from a person should discover memories about them (1 hop)."""
        graph = _build_graph()
        results = bfs_traverse(graph, seed_ids={"person-alice"}, max_hops=1)

        discovered_ids = {r.node_id for r in results}
        # Should find memories about Alice
        assert "mem-alice-cats" in discovered_ids
        assert "mem-alice-work" in discovered_ids
        assert "mem-bob-alice" in discovered_ids
        # Should find merged person
        assert "person-dup" in discovered_ids

    def test_two_hop_discovers_related_persons(self):
        """2-hop BFS from person-alice should discover person-bob."""
        graph = _build_graph()
        results = bfs_traverse(graph, seed_ids={"person-alice"}, max_hops=2)

        discovered_ids = {r.node_id for r in results}
        # Hop 1: memories about Alice + person-dup (via MERGED_INTO)
        assert "mem-alice-cats" in discovered_ids
        assert "mem-alice-work" in discovered_ids
        assert "mem-bob-alice" in discovered_ids
        assert "person-dup" in discovered_ids
        # Hop 2: person-bob (via mem-bob-alice → person-bob)
        assert "person-bob" in discovered_ids
        # mem-bob-music is 3 hops away, NOT discovered at max_hops=2
        assert "mem-bob-music" not in discovered_ids

    def test_three_hop_discovers_distant_memories(self):
        """3-hop BFS from person-alice should discover Bob's memories."""
        graph = _build_graph()
        results = bfs_traverse(graph, seed_ids={"person-alice"}, max_hops=3)

        discovered_ids = {r.node_id for r in results}
        assert "mem-bob-music" in discovered_ids

    def test_supersedes_excluded_by_default(self):
        """SUPERSEDES edges should be excluded by default."""
        graph = _build_graph()
        results = bfs_traverse(graph, seed_ids={"mem-new"}, max_hops=2)

        discovered_ids = {r.node_id for r in results}
        # Should NOT follow supersession chain
        assert "mem-old" not in discovered_ids

    def test_custom_exclude_types(self):
        """Can override excluded edge types."""
        graph = _build_graph()
        # Exclude ABOUT but allow SUPERSEDES
        results = bfs_traverse(
            graph,
            seed_ids={"mem-new"},
            max_hops=2,
            exclude_edge_types={ABOUT},
        )

        discovered_ids = {r.node_id for r in results}
        # Should follow SUPERSEDES now
        assert "mem-old" in discovered_ids

    def test_hops_tracked_correctly(self):
        """Each result should have correct hop count."""
        graph = _build_graph()
        results = bfs_traverse(graph, seed_ids={"person-alice"}, max_hops=2)

        by_id = {r.node_id: r for r in results}
        # Memories directly about Alice are 1 hop
        assert by_id["mem-alice-cats"].hops == 1
        assert by_id["mem-alice-work"].hops == 1
        assert by_id["mem-bob-alice"].hops == 1
        # Bob (reached via mem-bob-alice) is 2 hops from Alice
        assert by_id["person-bob"].hops == 2

    def test_path_tracked_correctly(self):
        """Each result should have the edge IDs traversed."""
        graph = _build_graph()
        results = bfs_traverse(graph, seed_ids={"person-alice"}, max_hops=2)

        for r in results:
            assert len(r.path) == r.hops

    def test_seeds_not_in_results(self):
        """Seed nodes should not appear in results."""
        graph = _build_graph()
        results = bfs_traverse(graph, seed_ids={"person-alice"}, max_hops=2)

        discovered_ids = {r.node_id for r in results}
        assert "person-alice" not in discovered_ids

    def test_empty_seeds(self):
        """Empty seed set returns empty results."""
        graph = _build_graph()
        results = bfs_traverse(graph, seed_ids=set(), max_hops=2)
        assert results == []

    def test_max_hops_zero(self):
        """max_hops=0 returns nothing (no expansion)."""
        graph = _build_graph()
        results = bfs_traverse(graph, seed_ids={"person-alice"}, max_hops=0)
        assert results == []

    def test_filter_fn(self):
        """filter_fn can reject certain nodes."""
        graph = _build_graph()

        # Only allow memory nodes
        def only_memories(node_id: str, edge) -> bool:
            return graph._node_type.get(node_id) == "memory"

        results = bfs_traverse(
            graph,
            seed_ids={"person-alice"},
            max_hops=2,
            filter_fn=only_memories,
        )

        for r in results:
            assert r.node_type == "memory"

    def test_multiple_seeds(self):
        """BFS from multiple seeds discovers union of reachable nodes."""
        graph = _build_graph()
        results = bfs_traverse(
            graph,
            seed_ids={"person-alice", "person-bob"},
            max_hops=1,
        )

        discovered_ids = {r.node_id for r in results}
        assert "mem-alice-cats" in discovered_ids
        assert "mem-bob-music" in discovered_ids
        assert "mem-bob-alice" in discovered_ids

    def test_no_duplicates_in_results(self):
        """Each node should appear at most once in results."""
        graph = _build_graph()
        results = bfs_traverse(
            graph,
            seed_ids={"person-alice", "person-bob"},
            max_hops=2,
        )

        node_ids = [r.node_id for r in results]
        assert len(node_ids) == len(set(node_ids))


class TestBFSNodeTypes:
    def test_node_types_correct(self):
        """Discovered nodes should have correct type annotations."""
        graph = _build_graph()
        results = bfs_traverse(graph, seed_ids={"person-alice"}, max_hops=2)

        by_id = {r.node_id: r for r in results}

        for nid, r in by_id.items():
            if nid.startswith("mem-"):
                assert r.node_type == "memory"
            elif nid.startswith("person-"):
                assert r.node_type == "person"
