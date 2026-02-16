"""Tests for graph edge infrastructure: CRUD, adjacency, query helpers, backfill."""

from __future__ import annotations

from datetime import UTC, datetime

from ash.graph.edges import (
    ABOUT,
    IS_PERSON,
    MERGED_INTO,
    SUPERSEDES,
    create_about_edge,
    create_is_person_edge,
    create_merged_into_edge,
    create_supersedes_edge,
    follow_merge_chain,
    get_memories_about_person,
    get_merged_into,
    get_person_for_user,
    get_subject_person_ids,
    get_subject_person_ids_batch,
    get_superseded_by,
    get_supersession_targets,
    get_users_for_person,
)
from ash.graph.graph import Edge, KnowledgeGraph
from ash.store.store import Store
from ash.store.types import MemoryEntry, MemoryType, PersonEntry, UserEntry

# =============================================================================
# Edge Creation Factories
# =============================================================================


class TestEdgeFactories:
    def test_create_about_edge(self):
        edge = create_about_edge("mem-1", "person-1")
        assert edge.edge_type == ABOUT
        assert edge.source_type == "memory"
        assert edge.source_id == "mem-1"
        assert edge.target_type == "person"
        assert edge.target_id == "person-1"
        assert edge.id.startswith("e-")
        assert edge.created_at is not None

    def test_create_about_edge_with_created_by(self):
        edge = create_about_edge("mem-1", "person-1", created_by="extraction")
        assert edge.created_by == "extraction"

    def test_create_supersedes_edge(self):
        edge = create_supersedes_edge("new-mem", "old-mem")
        assert edge.edge_type == SUPERSEDES
        assert edge.source_type == "memory"
        assert edge.source_id == "new-mem"
        assert edge.target_type == "memory"
        assert edge.target_id == "old-mem"

    def test_create_is_person_edge(self):
        edge = create_is_person_edge("user-1", "person-1")
        assert edge.edge_type == IS_PERSON
        assert edge.source_type == "user"
        assert edge.source_id == "user-1"
        assert edge.target_type == "person"
        assert edge.target_id == "person-1"

    def test_create_merged_into_edge(self):
        edge = create_merged_into_edge("person-dup", "person-primary")
        assert edge.edge_type == MERGED_INTO
        assert edge.source_type == "person"
        assert edge.source_id == "person-dup"
        assert edge.target_type == "person"
        assert edge.target_id == "person-primary"


# =============================================================================
# KnowledgeGraph Edge Operations
# =============================================================================


class TestGraphEdgeOps:
    def test_add_edge(self):
        graph = KnowledgeGraph()
        edge = create_about_edge("mem-1", "person-1")
        graph.add_edge(edge)

        assert edge.id in graph.edges
        assert edge.id in graph._outgoing["mem-1"]
        assert edge.id in graph._incoming["person-1"]
        assert edge.id in graph._edges_by_type[ABOUT]

    def test_remove_edge(self):
        graph = KnowledgeGraph()
        edge = create_about_edge("mem-1", "person-1")
        graph.add_edge(edge)
        graph.remove_edge(edge.id)

        assert edge.id not in graph.edges
        assert edge.id not in graph._outgoing.get("mem-1", [])
        assert edge.id not in graph._incoming.get("person-1", [])
        assert edge.id not in graph._edges_by_type.get(ABOUT, [])

    def test_remove_nonexistent_edge(self):
        graph = KnowledgeGraph()
        # Should not raise
        graph.remove_edge("nonexistent")

    def test_invalidate_edge(self):
        graph = KnowledgeGraph()
        edge = create_about_edge("mem-1", "person-1")
        graph.add_edge(edge)
        now = datetime.now(UTC)
        graph.invalidate_edge(edge.id, now)

        assert graph.edges[edge.id].invalid_at == now

    def test_get_outgoing(self):
        graph = KnowledgeGraph()
        e1 = create_about_edge("mem-1", "person-1")
        e2 = create_about_edge("mem-1", "person-2")
        e3 = create_supersedes_edge("mem-1", "mem-old")
        graph.add_edge(e1)
        graph.add_edge(e2)
        graph.add_edge(e3)

        # All outgoing from mem-1
        all_out = graph.get_outgoing("mem-1")
        assert len(all_out) == 3

        # Filter by type
        about_out = graph.get_outgoing("mem-1", edge_type=ABOUT)
        assert len(about_out) == 2
        assert {e.target_id for e in about_out} == {"person-1", "person-2"}

    def test_get_incoming(self):
        graph = KnowledgeGraph()
        e1 = create_about_edge("mem-1", "person-1")
        e2 = create_about_edge("mem-2", "person-1")
        graph.add_edge(e1)
        graph.add_edge(e2)

        incoming = graph.get_incoming("person-1", edge_type=ABOUT)
        assert len(incoming) == 2
        assert {e.source_id for e in incoming} == {"mem-1", "mem-2"}

    def test_get_outgoing_active_only(self):
        graph = KnowledgeGraph()
        e1 = create_about_edge("mem-1", "person-1")
        e2 = create_about_edge("mem-1", "person-2")
        graph.add_edge(e1)
        graph.add_edge(e2)

        # Invalidate one
        graph.invalidate_edge(e2.id, datetime.now(UTC))

        active = graph.get_outgoing("mem-1", active_only=True)
        assert len(active) == 1
        assert active[0].target_id == "person-1"

        all_edges = graph.get_outgoing("mem-1", active_only=False)
        assert len(all_edges) == 2


# =============================================================================
# Edge Query Helpers
# =============================================================================


class TestEdgeQueryHelpers:
    def _build_graph(self) -> KnowledgeGraph:
        """Build a test graph with nodes and edges."""
        graph = KnowledgeGraph()

        # Add people
        for pid in ["person-alice", "person-bob", "person-dup"]:
            graph.add_person(
                PersonEntry(id=pid, name=pid.replace("person-", "").title())
            )

        # Add users
        graph.add_user(
            UserEntry(
                id="user-alice",
                provider="test",
                provider_id="alice",
            )
        )

        # Add memories
        for mid in ["mem-1", "mem-2", "mem-3", "mem-old"]:
            graph.add_memory(
                MemoryEntry(
                    id=mid,
                    content=f"Content of {mid}",
                    memory_type=MemoryType.KNOWLEDGE,
                )
            )

        # Add edges
        graph.add_edge(create_about_edge("mem-1", "person-alice"))
        graph.add_edge(create_about_edge("mem-1", "person-bob"))
        graph.add_edge(create_about_edge("mem-2", "person-alice"))
        graph.add_edge(create_supersedes_edge("mem-3", "mem-old"))
        graph.add_edge(create_is_person_edge("user-alice", "person-alice"))
        graph.add_edge(create_merged_into_edge("person-dup", "person-alice"))

        return graph

    def test_get_subject_person_ids(self):
        graph = self._build_graph()
        subjects = get_subject_person_ids(graph, "mem-1")
        assert set(subjects) == {"person-alice", "person-bob"}

    def test_get_subject_person_ids_empty(self):
        graph = self._build_graph()
        subjects = get_subject_person_ids(graph, "mem-3")
        assert subjects == []

    def test_get_subject_person_ids_batch(self):
        graph = self._build_graph()
        batch = get_subject_person_ids_batch(graph, ["mem-1", "mem-2", "mem-3"])
        assert set(batch["mem-1"]) == {"person-alice", "person-bob"}
        assert batch["mem-2"] == ["person-alice"]
        assert "mem-3" not in batch  # No subjects

    def test_get_memories_about_person(self):
        graph = self._build_graph()
        memories = get_memories_about_person(graph, "person-alice")
        assert set(memories) == {"mem-1", "mem-2"}

    def test_get_memories_about_person_empty(self):
        graph = self._build_graph()
        memories = get_memories_about_person(graph, "nonexistent")
        assert memories == []

    def test_get_superseded_by(self):
        graph = self._build_graph()
        # mem-old was superseded by mem-3
        superseder = get_superseded_by(graph, "mem-old")
        assert superseder == "mem-3"

    def test_get_superseded_by_none(self):
        graph = self._build_graph()
        superseder = get_superseded_by(graph, "mem-1")
        assert superseder is None

    def test_get_supersession_targets(self):
        graph = self._build_graph()
        # mem-3 supersedes mem-old
        targets = get_supersession_targets(graph, "mem-3")
        assert targets == ["mem-old"]

    def test_get_person_for_user(self):
        graph = self._build_graph()
        person_id = get_person_for_user(graph, "user-alice")
        assert person_id == "person-alice"

    def test_get_person_for_user_none(self):
        graph = self._build_graph()
        person_id = get_person_for_user(graph, "nonexistent")
        assert person_id is None

    def test_get_users_for_person(self):
        graph = self._build_graph()
        user_ids = get_users_for_person(graph, "person-alice")
        assert user_ids == ["user-alice"]

    def test_get_merged_into(self):
        graph = self._build_graph()
        target = get_merged_into(graph, "person-dup")
        assert target == "person-alice"

    def test_get_merged_into_none(self):
        graph = self._build_graph()
        target = get_merged_into(graph, "person-alice")
        assert target is None

    def test_follow_merge_chain(self):
        graph = KnowledgeGraph()
        for pid in ["p1", "p2", "p3"]:
            graph.add_person(PersonEntry(id=pid, name=pid))
        graph.add_edge(create_merged_into_edge("p1", "p2"))
        graph.add_edge(create_merged_into_edge("p2", "p3"))

        final = follow_merge_chain(graph, "p1")
        assert final == "p3"

    def test_follow_merge_chain_no_merges(self):
        graph = KnowledgeGraph()
        graph.add_person(PersonEntry(id="p1", name="P1"))

        final = follow_merge_chain(graph, "p1")
        assert final == "p1"

    def test_follow_merge_chain_max_depth(self):
        graph = KnowledgeGraph()
        # Create a chain longer than max_depth
        for i in range(15):
            graph.add_person(PersonEntry(id=f"p{i}", name=f"P{i}"))
        for i in range(14):
            graph.add_edge(create_merged_into_edge(f"p{i}", f"p{i + 1}"))

        # With max_depth=5, should stop early
        final = follow_merge_chain(graph, "p0", max_depth=5)
        assert final == "p5"


# =============================================================================
# Edge Serialization
# =============================================================================


class TestEdgeSerialization:
    def test_edge_to_dict(self):
        edge = Edge(
            id="e-123",
            edge_type=ABOUT,
            source_type="memory",
            source_id="mem-1",
            target_type="person",
            target_id="person-1",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        d = edge.to_dict()
        assert d["id"] == "e-123"
        assert d["edge_type"] == "ABOUT"
        assert d["source_id"] == "mem-1"
        assert d["target_id"] == "person-1"
        assert "created_at" in d

    def test_edge_to_dict_omits_defaults(self):
        edge = Edge(
            id="e-123",
            edge_type=ABOUT,
            source_type="memory",
            source_id="mem-1",
            target_type="person",
            target_id="person-1",
        )
        d = edge.to_dict()
        assert "weight" not in d
        assert "properties" not in d
        assert "episode_id" not in d

    def test_edge_roundtrip(self):
        edge = Edge(
            id="e-123",
            edge_type=ABOUT,
            source_type="memory",
            source_id="mem-1",
            target_type="person",
            target_id="person-1",
            weight=0.5,
            properties={"key": "value"},
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            created_by="test",
        )
        d = edge.to_dict()
        restored = Edge.from_dict(d)

        assert restored.id == edge.id
        assert restored.edge_type == edge.edge_type
        assert restored.source_id == edge.source_id
        assert restored.target_id == edge.target_id
        assert restored.weight == edge.weight
        assert restored.properties == edge.properties
        assert restored.created_by == edge.created_by


# =============================================================================
# Dual-Write Integration Tests
# =============================================================================


class TestDualWriteEdges:
    """Test that Store operations create both FK fields and edges."""

    async def test_add_memory_creates_about_edges(self, graph_store: Store):
        person = await graph_store.create_person(
            created_by="test", name="Alice", aliases=["alice"]
        )
        memory = await graph_store.add_memory(
            content="Alice likes cats",
            subject_person_ids=[person.id],
        )

        # Verify ABOUT edge exists
        about_edges = graph_store._graph.get_outgoing(memory.id, edge_type=ABOUT)
        assert len(about_edges) == 1
        assert about_edges[0].target_id == person.id

        # Verify edge query helper works
        subjects = get_subject_person_ids(graph_store._graph, memory.id)
        assert person.id in subjects

    async def test_supersede_creates_supersedes_edge(self, graph_store: Store):
        old_mem = await graph_store.add_memory(content="Bob likes blue")
        new_mem = await graph_store.add_memory(content="Bob likes red")

        await graph_store._mark_superseded(old_mem.id, new_mem.id)

        # Verify SUPERSEDES edge
        targets = get_supersession_targets(graph_store._graph, new_mem.id)
        assert old_mem.id in targets

        superseder = get_superseded_by(graph_store._graph, old_mem.id)
        assert superseder == new_mem.id

    async def test_ensure_user_creates_is_person_edge(self, graph_store: Store):
        person = await graph_store.create_person(
            created_by="test", name="Alice", aliases=["alice"]
        )
        user = await graph_store.ensure_user(
            provider="test",
            provider_id="alice123",
            username="alice",
            person_id=person.id,
        )

        # Verify IS_PERSON edge
        pid = get_person_for_user(graph_store._graph, user.id)
        assert pid == person.id

        user_ids = get_users_for_person(graph_store._graph, person.id)
        assert user.id in user_ids

    async def test_merge_people_creates_merged_into_edge(self, graph_store: Store):
        primary = await graph_store.create_person(
            created_by="test", name="Alice", aliases=["alice"]
        )
        secondary = await graph_store.create_person(
            created_by="test", name="Ali", aliases=["ali"]
        )

        await graph_store.merge_people(primary.id, secondary.id)

        # Verify MERGED_INTO edge
        target = get_merged_into(graph_store._graph, secondary.id)
        assert target == primary.id

    async def test_batch_update_syncs_about_edges(self, graph_store: Store):
        person_a = await graph_store.create_person(
            created_by="test", name="Alice", aliases=["alice"]
        )
        person_b = await graph_store.create_person(
            created_by="test", name="Bob", aliases=["bob"]
        )
        memory = await graph_store.add_memory(
            content="About Alice",
            subject_person_ids=[person_a.id],
        )

        # Update to be about Bob instead via subject_person_ids_map
        await graph_store.batch_update_memories(
            [memory], subject_person_ids_map={memory.id: [person_b.id]}
        )

        # Verify edges were synced
        subjects = get_subject_person_ids(graph_store._graph, memory.id)
        assert person_b.id in subjects
        assert person_a.id not in subjects

    async def test_remap_subject_updates_about_edges(self, graph_store: Store):
        old_person = await graph_store.create_person(
            created_by="test", name="Alice", aliases=["alice"]
        )
        new_person = await graph_store.create_person(
            created_by="test", name="Alice Primary", aliases=["alice-primary"]
        )
        memory = await graph_store.add_memory(
            content="About Alice",
            subject_person_ids=[old_person.id],
        )

        await graph_store.remap_subject_person_id(old_person.id, new_person.id)

        # Verify ABOUT edges were remapped
        subjects = get_subject_person_ids(graph_store._graph, memory.id)
        assert new_person.id in subjects
        assert old_person.id not in subjects


# =============================================================================
# Edge-Based Read Path Tests
# =============================================================================


class TestEdgeBasedReads:
    """Test that read operations use edges correctly."""

    async def test_memories_about_person_uses_edges(self, graph_store: Store):
        person = await graph_store.create_person(
            created_by="test", name="Alice", aliases=["alice"]
        )
        mem1 = await graph_store.add_memory(
            content="Alice likes cats",
            subject_person_ids=[person.id],
        )
        mem2 = await graph_store.add_memory(
            content="Alice works at Acme",
            subject_person_ids=[person.id],
        )
        await graph_store.add_memory(content="Unrelated memory")

        result = await graph_store.memories_about_person(person.id)
        assert mem1.id in result
        assert mem2.id in result
        assert len(result) == 2

    async def test_forget_person_uses_edges(self, graph_store: Store):
        person = await graph_store.create_person(
            created_by="test", name="Alice", aliases=["alice"]
        )
        mem = await graph_store.add_memory(
            content="Alice likes cats",
            subject_person_ids=[person.id],
        )

        count = await graph_store.forget_person(person.id)
        assert count == 1

        # Verify memory was archived
        memory = graph_store._graph.memories.get(mem.id)
        assert memory is not None
        assert memory.archived_at is not None

    async def test_users_for_person_uses_edges(self, graph_store: Store):
        person = await graph_store.create_person(
            created_by="test", name="Alice", aliases=["alice"]
        )
        user = await graph_store.ensure_user(
            provider="test",
            provider_id="alice123",
            username="alice",
            person_id=person.id,
        )

        result = await graph_store.users_for_person(person.id)
        assert len(result) == 1
        assert result[0].id == user.id

    async def test_follow_merge_chain_uses_edges(self, graph_store: Store):
        primary = await graph_store.create_person(
            created_by="test", name="Alice", aliases=["alice"]
        )
        secondary = await graph_store.create_person(
            created_by="test", name="Ali", aliases=["ali"]
        )

        await graph_store.merge_people(primary.id, secondary.id)

        # Follow chain should resolve secondary → primary
        result = await graph_store._follow_merge_chain(secondary)
        assert result.id == primary.id

    async def test_supersession_chain_uses_edges(self, graph_store: Store):
        mem1 = await graph_store.add_memory(content="Version 1")
        mem2 = await graph_store.add_memory(content="Version 2")
        mem3 = await graph_store.add_memory(content="Version 3")

        await graph_store._mark_superseded(mem1.id, mem2.id)
        await graph_store._mark_superseded(mem2.id, mem3.id)

        chain = await graph_store.get_supersession_chain(mem3.id)
        chain_ids = [m.id for m in chain]
        assert mem2.id in chain_ids


# =============================================================================
# STATED_BY Edge Tests
# =============================================================================


class TestStatedByEdges:
    """Tests for STATED_BY edge creation and query helpers."""

    def test_create_stated_by_edge(self):
        from ash.graph.edges import STATED_BY, create_stated_by_edge

        edge = create_stated_by_edge("mem-1", "person-1")
        assert edge.edge_type == STATED_BY
        assert edge.source_type == "memory"
        assert edge.source_id == "mem-1"
        assert edge.target_type == "person"
        assert edge.target_id == "person-1"

    def test_create_stated_by_edge_with_created_by(self):
        from ash.graph.edges import create_stated_by_edge

        edge = create_stated_by_edge("mem-1", "person-1", created_by="extraction")
        assert edge.created_by == "extraction"

    def test_get_stated_by_person(self):
        from ash.graph.edges import create_stated_by_edge, get_stated_by_person

        graph = KnowledgeGraph()
        graph.add_memory(
            MemoryEntry(id="mem-1", content="test", memory_type=MemoryType.KNOWLEDGE)
        )
        graph.add_person(PersonEntry(id="person-1", name="Alice"))
        graph.add_edge(create_stated_by_edge("mem-1", "person-1"))

        result = get_stated_by_person(graph, "mem-1")
        assert result == "person-1"

    def test_get_stated_by_person_none(self):
        from ash.graph.edges import get_stated_by_person

        graph = KnowledgeGraph()
        graph.add_memory(
            MemoryEntry(id="mem-1", content="test", memory_type=MemoryType.KNOWLEDGE)
        )
        assert get_stated_by_person(graph, "mem-1") is None

    def test_get_memories_stated_by(self):
        from ash.graph.edges import create_stated_by_edge, get_memories_stated_by

        graph = KnowledgeGraph()
        for mid in ["mem-1", "mem-2", "mem-3"]:
            graph.add_memory(
                MemoryEntry(
                    id=mid, content=f"test {mid}", memory_type=MemoryType.KNOWLEDGE
                )
            )
        graph.add_person(PersonEntry(id="person-1", name="Alice"))
        graph.add_edge(create_stated_by_edge("mem-1", "person-1"))
        graph.add_edge(create_stated_by_edge("mem-2", "person-1"))

        result = get_memories_stated_by(graph, "person-1")
        assert set(result) == {"mem-1", "mem-2"}

    async def test_add_memory_creates_stated_by_edge(self, graph_store: Store):
        from ash.graph.edges import STATED_BY

        person = await graph_store.create_person(
            created_by="test", name="Alice", aliases=["alice"]
        )
        memory = await graph_store.add_memory(
            content="Alice said something",
            source_username="alice",
            stated_by_person_id=person.id,
        )

        stated_by_edges = graph_store._graph.get_outgoing(
            memory.id, edge_type=STATED_BY
        )
        assert len(stated_by_edges) == 1
        assert stated_by_edges[0].target_id == person.id

    async def test_add_memory_no_stated_by_without_person(self, graph_store: Store):
        from ash.graph.edges import STATED_BY

        memory = await graph_store.add_memory(
            content="Nobody said this",
        )

        stated_by_edges = graph_store._graph.get_outgoing(
            memory.id, edge_type=STATED_BY
        )
        assert len(stated_by_edges) == 0


# =============================================================================
# HAS_RELATIONSHIP Edge Tests
# =============================================================================


class TestHasRelationshipEdges:
    """Tests for HAS_RELATIONSHIP edge creation and query helpers."""

    def test_create_has_relationship_edge(self):
        from ash.graph.edges import HAS_RELATIONSHIP, create_has_relationship_edge

        edge = create_has_relationship_edge(
            "person-1", "person-2", relationship_type="friend", stated_by="alice"
        )
        assert edge.edge_type == HAS_RELATIONSHIP
        assert edge.source_type == "person"
        assert edge.source_id == "person-1"
        assert edge.target_type == "person"
        assert edge.target_id == "person-2"
        assert edge.properties == {"relationship_type": "friend", "stated_by": "alice"}

    def test_create_has_relationship_edge_no_properties(self):
        from ash.graph.edges import create_has_relationship_edge

        edge = create_has_relationship_edge("person-1", "person-2")
        assert edge.properties is None

    def test_get_relationships_for_person(self):
        from ash.graph.edges import (
            create_has_relationship_edge,
            get_relationships_for_person,
        )

        graph = KnowledgeGraph()
        graph.add_person(PersonEntry(id="p1", name="Alice"))
        graph.add_person(PersonEntry(id="p2", name="Bob"))
        graph.add_person(PersonEntry(id="p3", name="Charlie"))
        graph.add_edge(
            create_has_relationship_edge("p1", "p2", relationship_type="friend")
        )
        graph.add_edge(
            create_has_relationship_edge("p3", "p1", relationship_type="colleague")
        )

        edges = get_relationships_for_person(graph, "p1")
        assert len(edges) == 2

    def test_get_related_people(self):
        from ash.graph.edges import create_has_relationship_edge, get_related_people

        graph = KnowledgeGraph()
        graph.add_person(PersonEntry(id="p1", name="Alice"))
        graph.add_person(PersonEntry(id="p2", name="Bob"))
        graph.add_person(PersonEntry(id="p3", name="Charlie"))
        graph.add_edge(create_has_relationship_edge("p1", "p2"))
        graph.add_edge(create_has_relationship_edge("p3", "p1"))

        related = get_related_people(graph, "p1")
        assert set(related) == {"p2", "p3"}

    async def test_add_relationship_creates_edge(self, graph_store: Store):
        from ash.graph.edges import HAS_RELATIONSHIP

        alice = await graph_store.create_person(
            created_by="test", name="Alice", aliases=["alice"]
        )
        bob = await graph_store.create_person(
            created_by="test", name="Bob", aliases=["bob"]
        )

        await graph_store.add_relationship(
            bob.id, "friend", stated_by="alice", related_person_id=alice.id
        )

        edges = graph_store._graph.get_outgoing(bob.id, edge_type=HAS_RELATIONSHIP)
        assert len(edges) == 1
        assert edges[0].target_id == alice.id
        assert edges[0].properties is not None
        assert edges[0].properties["relationship_type"] == "friend"

    async def test_add_relationship_no_edge_without_related_person(
        self, graph_store: Store
    ):
        from ash.graph.edges import HAS_RELATIONSHIP

        bob = await graph_store.create_person(
            created_by="test", name="Bob", aliases=["bob"]
        )

        await graph_store.add_relationship(bob.id, "friend", stated_by="alice")

        edges = graph_store._graph.get_outgoing(bob.id, edge_type=HAS_RELATIONSHIP)
        assert len(edges) == 0


# =============================================================================
# Edge Cascade Tests (remove_edges_for_node)
# =============================================================================


class TestEdgeCascade:
    """Tests for cascading edge removal when nodes are deleted."""

    def test_remove_memory_cascades_edges(self):
        from ash.graph.edges import create_stated_by_edge

        graph = KnowledgeGraph()
        graph.add_person(PersonEntry(id="person-1", name="Alice"))
        graph.add_memory(
            MemoryEntry(id="mem-1", content="test", memory_type=MemoryType.KNOWLEDGE)
        )
        graph.add_edge(create_about_edge("mem-1", "person-1"))
        graph.add_edge(create_stated_by_edge("mem-1", "person-1"))

        assert len(graph.edges) == 2

        graph.remove_memory("mem-1")

        assert len(graph.edges) == 0
        assert graph.get_outgoing("mem-1", edge_type=ABOUT) == []
        assert graph.get_incoming("person-1", edge_type=ABOUT) == []

    def test_remove_person_cascades_edges(self):
        from ash.graph.edges import (
            HAS_RELATIONSHIP,
            create_has_relationship_edge,
        )

        graph = KnowledgeGraph()
        graph.add_person(PersonEntry(id="person-1", name="Alice"))
        graph.add_person(PersonEntry(id="person-2", name="Bob"))
        graph.add_user(UserEntry(id="user-1", provider="test", provider_id="alice"))
        graph.add_memory(
            MemoryEntry(id="mem-1", content="test", memory_type=MemoryType.KNOWLEDGE)
        )

        graph.add_edge(create_about_edge("mem-1", "person-1"))
        graph.add_edge(create_is_person_edge("user-1", "person-1"))
        graph.add_edge(
            create_has_relationship_edge(
                "person-1", "person-2", relationship_type="friend"
            )
        )

        assert len(graph.edges) == 3

        graph.remove_person("person-1")

        assert len(graph.edges) == 0
        assert graph.get_incoming("person-1", edge_type=ABOUT) == []
        assert graph.get_outgoing("user-1", edge_type=IS_PERSON) == []
        assert graph.get_outgoing("person-1", edge_type=HAS_RELATIONSHIP) == []

    def test_remove_edges_for_node_no_edges(self):
        graph = KnowledgeGraph()
        graph.add_memory(
            MemoryEntry(id="mem-1", content="test", memory_type=MemoryType.KNOWLEDGE)
        )

        removed = graph.remove_edges_for_node("mem-1")
        assert removed == []

    def test_remove_edges_for_node_nonexistent(self):
        graph = KnowledgeGraph()
        removed = graph.remove_edges_for_node("nonexistent")
        assert removed == []


# =============================================================================
# Store Integration: Cascade + Persistence
# =============================================================================


class TestStoreCascadeIntegration:
    """Tests that store operations cascade edges and persist correctly."""

    async def test_compact_removes_edges(self, graph_store: Store):
        """Compact should remove edges for archived memories."""

        person = await graph_store.create_person(
            created_by="test", name="Alice", aliases=["alice"]
        )
        memory = await graph_store.add_memory(
            content="Alice fact",
            subject_person_ids=[person.id],
        )

        # Manually archive the memory with old timestamp
        memory.archived_at = datetime(2020, 1, 1, tzinfo=UTC)

        count = await graph_store.compact(older_than_days=1)
        assert count == 1

        # Edges should be gone
        about_edges = graph_store._graph.get_incoming(person.id, edge_type=ABOUT)
        assert len(about_edges) == 0

    async def test_clear_removes_memory_edges(self, graph_store: Store):
        """Clear should remove all memory edges."""
        person = await graph_store.create_person(
            created_by="test", name="Alice", aliases=["alice"]
        )
        await graph_store.add_memory(
            content="Alice fact 1",
            subject_person_ids=[person.id],
        )
        await graph_store.add_memory(
            content="Alice fact 2",
            subject_person_ids=[person.id],
        )

        about_before = graph_store._graph.get_incoming(person.id, edge_type=ABOUT)
        assert len(about_before) == 2

        await graph_store.clear()

        about_after = graph_store._graph.get_incoming(person.id, edge_type=ABOUT)
        assert len(about_after) == 0
        assert len(graph_store._graph.memories) == 0

    async def test_delete_person_cascades_all_edges(self, graph_store: Store):
        """Deleting a person should remove all connected edges."""
        from ash.graph.edges import HAS_RELATIONSHIP

        alice = await graph_store.create_person(
            created_by="test", name="Alice", aliases=["alice"]
        )
        bob = await graph_store.create_person(
            created_by="test", name="Bob", aliases=["bob"]
        )
        user = await graph_store.ensure_user(
            provider="test",
            provider_id="alice123",
            username="alice",
            person_id=alice.id,
        )
        await graph_store.add_memory(
            content="About Alice",
            subject_person_ids=[alice.id],
        )
        await graph_store.add_relationship(
            alice.id, "friend", stated_by="test", related_person_id=bob.id
        )

        # Verify edges exist
        assert len(graph_store._graph.get_incoming(alice.id, edge_type=ABOUT)) > 0
        assert len(graph_store._graph.get_outgoing(user.id, edge_type=IS_PERSON)) > 0

        await graph_store.delete_person(alice.id)

        # All edges to/from alice should be gone
        assert graph_store._graph.get_incoming(alice.id, edge_type=ABOUT) == []
        assert graph_store._graph.get_incoming(alice.id, edge_type=IS_PERSON) == []
        assert (
            graph_store._graph.get_outgoing(alice.id, edge_type=HAS_RELATIONSHIP) == []
        )
