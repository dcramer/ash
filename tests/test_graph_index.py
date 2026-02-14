"""Tests for GraphIndex."""

from datetime import UTC, datetime
from typing import Any

from ash.graph.index import GraphIndex
from ash.graph.types import ChatEntry, EdgeType, UserEntry
from ash.memory.types import MemoryEntry, MemoryType
from ash.people.types import PersonEntry, RelationshipClaim


def _make_memory(**kwargs: Any) -> MemoryEntry:
    defaults: dict[str, Any] = {
        "id": "m1",
        "content": "test",
        "memory_type": MemoryType.KNOWLEDGE,
        "created_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    return MemoryEntry(**defaults)


def _make_person(**kwargs: Any) -> PersonEntry:
    defaults: dict[str, Any] = {
        "id": "p1",
        "name": "Test Person",
        "created_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    return PersonEntry(**defaults)


def _make_user(**kwargs: Any) -> UserEntry:
    defaults: dict[str, Any] = {
        "id": "u1",
        "provider": "telegram",
        "provider_id": "100",
        "created_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    return UserEntry(**defaults)


def _make_chat(**kwargs: Any) -> ChatEntry:
    defaults: dict[str, Any] = {
        "id": "c1",
        "provider": "telegram",
        "provider_id": "200",
        "created_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    return ChatEntry(**defaults)


class TestGraphIndexBuild:
    def test_empty_build(self):
        idx = GraphIndex()
        idx.build([], [], [], [])
        assert idx.memories_about("p1") == set()

    def test_about_edges(self):
        memory = _make_memory(id="m1", subject_person_ids=["p1", "p2"])
        person1 = _make_person(id="p1")
        person2 = _make_person(id="p2")

        idx = GraphIndex()
        idx.build([memory], [person1, person2], [], [])

        assert idx.memories_about("p1") == {"m1"}
        assert idx.memories_about("p2") == {"m1"}
        assert idx.neighbors("m1", EdgeType.ABOUT) == {"p1", "p2"}

    def test_owned_by_edges(self):
        user = _make_user(id="u1", provider_id="100")
        memory = _make_memory(id="m1", owner_user_id="100")

        idx = GraphIndex()
        idx.build([memory], [], [user], [])

        assert idx.neighbors("m1", EdgeType.OWNED_BY) == {"u1"}
        assert idx.neighbors("u1", EdgeType.OWNED_BY, "incoming") == {"m1"}

    def test_in_chat_edges(self):
        chat = _make_chat(id="c1", provider_id="200")
        memory = _make_memory(id="m1", chat_id="200")

        idx = GraphIndex()
        idx.build([memory], [], [], [chat])

        assert idx.neighbors("m1", EdgeType.IN_CHAT) == {"c1"}

    def test_stated_by_edges(self):
        user = _make_user(id="u1", username="notzeeg")
        memory = _make_memory(id="m1", source_username="notzeeg")

        idx = GraphIndex()
        idx.build([memory], [], [user], [])

        assert idx.neighbors("m1", EdgeType.STATED_BY) == {"u1"}

    def test_stated_by_case_insensitive(self):
        user = _make_user(id="u1", username="NotZeeg")
        memory = _make_memory(id="m1", source_username="notzeeg")

        idx = GraphIndex()
        idx.build([memory], [], [user], [])

        assert idx.neighbors("m1", EdgeType.STATED_BY) == {"u1"}

    def test_supersedes_edges(self):
        m1 = _make_memory(id="m1", superseded_by_id="m2")
        m2 = _make_memory(id="m2")

        idx = GraphIndex()
        idx.build([m1, m2], [], [], [])

        assert idx.neighbors("m1", EdgeType.SUPERSEDES) == {"m2"}
        assert idx.neighbors("m2", EdgeType.SUPERSEDES, "incoming") == {"m1"}

    def test_knows_edges(self):
        user = _make_user(id="u1", username="notzeeg")
        person = _make_person(
            id="p1",
            relationships=[RelationshipClaim(relationship="wife", stated_by="notzeeg")],
        )

        idx = GraphIndex()
        idx.build([], [person], [user], [])

        assert idx.neighbors("u1", EdgeType.KNOWS) == {"p1"}
        assert idx.people_known_by_user("u1") == {"p1"}

    def test_knows_via_provider_id(self):
        user = _make_user(id="u1", provider_id="100")
        person = _make_person(
            id="p1",
            relationships=[RelationshipClaim(relationship="self", stated_by="100")],
        )

        idx = GraphIndex()
        idx.build([], [person], [user], [])

        assert idx.neighbors("u1", EdgeType.KNOWS) == {"p1"}

    def test_is_person_edges(self):
        user = _make_user(id="u1", person_id="p1")
        person = _make_person(id="p1")

        idx = GraphIndex()
        idx.build([], [person], [user], [])

        assert idx.neighbors("u1", EdgeType.IS_PERSON) == {"p1"}
        assert idx.neighbors("p1", EdgeType.IS_PERSON, "incoming") == {"u1"}

    def test_merged_into_edges(self):
        p1 = _make_person(id="p1", merged_into="p2")
        p2 = _make_person(id="p2")

        idx = GraphIndex()
        idx.build([], [p1, p2], [], [])

        assert idx.neighbors("p1", EdgeType.MERGED_INTO) == {"p2"}

    def test_archived_memories_excluded(self):
        memory = _make_memory(
            id="m1",
            subject_person_ids=["p1"],
            archived_at=datetime.now(UTC),
        )
        person = _make_person(id="p1")

        idx = GraphIndex()
        idx.build([memory], [person], [], [])

        assert idx.memories_about("p1") == set()


class TestGraphIndexLookups:
    def test_resolve_user_by_provider_id(self):
        user = _make_user(id="u1", provider_id="100")

        idx = GraphIndex()
        idx.build([], [], [user], [])

        assert idx.resolve_user("100") == "u1"
        assert idx.resolve_user("999") is None

    def test_resolve_user_by_username(self):
        user = _make_user(id="u1", username="notzeeg")

        idx = GraphIndex()
        idx.build([], [], [user], [])

        assert idx.resolve_user_by_username("notzeeg") == "u1"
        assert idx.resolve_user_by_username("NotZeeg") == "u1"
        assert idx.resolve_user_by_username("unknown") is None

    def test_resolve_chat(self):
        chat = _make_chat(id="c1", provider_id="200")

        idx = GraphIndex()
        idx.build([], [], [], [chat])

        assert idx.resolve_chat("200") == "c1"
        assert idx.resolve_chat("999") is None


class TestGraphIndexTraversal:
    def test_single_hop(self):
        user = _make_user(id="u1", username="notzeeg")
        person = _make_person(
            id="p1",
            relationships=[RelationshipClaim(relationship="wife", stated_by="notzeeg")],
        )

        idx = GraphIndex()
        idx.build([], [person], [user], [])

        result = idx.traverse("u1", [EdgeType.KNOWS], max_hops=1)
        assert result == {"p1"}

    def test_multi_hop(self):
        user = _make_user(id="u1", username="notzeeg", person_id="p1")
        person = _make_person(
            id="p1",
            relationships=[RelationshipClaim(relationship="self", stated_by="notzeeg")],
        )
        memory = _make_memory(
            id="m1", subject_person_ids=["p1"], source_username="notzeeg"
        )

        idx = GraphIndex()
        idx.build([memory], [person], [user], [])

        # User -> Person (IS_PERSON), then Person -> Memory (reverse ABOUT)
        # traverse only goes outgoing, so we check neighbors directly
        persons = idx.traverse("u1", [EdgeType.IS_PERSON], max_hops=1)
        assert persons == {"p1"}

        memories = idx.memories_about("p1")
        assert memories == {"m1"}

    def test_traverse_max_hops_respected(self):
        user = _make_user(id="u1", username="notzeeg")
        p1 = _make_person(
            id="p1",
            relationships=[RelationshipClaim(relationship="wife", stated_by="notzeeg")],
        )
        p2 = _make_person(id="p2", merged_into="p3")
        p3 = _make_person(id="p3")

        idx = GraphIndex()
        idx.build([], [p1, p2, p3], [user], [])

        # With max_hops=1, should only get direct neighbors
        result = idx.traverse("u1", [EdgeType.KNOWS], max_hops=1)
        assert result == {"p1"}

    def test_traverse_empty_result(self):
        idx = GraphIndex()
        idx.build([], [], [], [])
        assert idx.traverse("nonexistent", [EdgeType.ABOUT], max_hops=2) == set()


class TestGraphIndexRebuild:
    def test_rebuild_clears_old_data(self):
        user = _make_user(id="u1", username="notzeeg")
        memory = _make_memory(id="m1", source_username="notzeeg")

        idx = GraphIndex()
        idx.build([memory], [], [user], [])
        assert idx.neighbors("m1", EdgeType.STATED_BY) == {"u1"}

        # Rebuild with no data
        idx.build([], [], [], [])
        assert idx.neighbors("m1", EdgeType.STATED_BY) == set()
        assert idx.resolve_user_by_username("notzeeg") is None
