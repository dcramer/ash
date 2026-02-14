"""Tests for GraphStore graph-based optimizations.

Verifies that GraphStore uses the GraphIndex for O(1) lookups
instead of linear scans over all memories.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.graph.store import GraphStore
from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.file_store import FileMemoryStore
from ash.memory.index import VectorIndex, VectorSearchResult
from ash.memory.types import Sensitivity


def _make_search_results(
    results: list[tuple[str, float]],
) -> list[VectorSearchResult]:
    return [VectorSearchResult(memory_id=mid, similarity=sim) for mid, sim in results]


@pytest.fixture
def mock_index():
    index = MagicMock(spec=VectorIndex)
    index.search = AsyncMock(return_value=[])
    index.add_embedding = AsyncMock()
    index.delete_embedding = AsyncMock()
    return index


@pytest.fixture
def mock_embedding_generator():
    generator = MagicMock(spec=EmbeddingGenerator)
    generator.embed = AsyncMock(return_value=[0.1] * 1536)
    return generator


@pytest.fixture
async def graph_store(
    ash_home: Path, mock_index, mock_embedding_generator
) -> GraphStore:
    store = FileMemoryStore()
    graph_dir = ash_home / "graph"
    graph_dir.mkdir(exist_ok=True)
    return GraphStore(
        memory_store=store,
        vector_index=mock_index,
        embedding_generator=mock_embedding_generator,
        people_path=graph_dir / "people.jsonl",
        users_path=graph_dir / "users.jsonl",
        chats_path=graph_dir / "chats.jsonl",
    )


class TestFindMemoriesAboutPersons:
    """Test graph-based memory lookup by subject person."""

    async def test_finds_memories_about_person(self, graph_store: GraphStore):
        """Graph-based lookup returns memories about a given person."""
        person = await graph_store.create_person(created_by="user-1", name="Bob")

        m1 = await graph_store._store.add_memory(
            content="Bob likes hiking",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        await graph_store._store.add_memory(
            content="Alice likes cooking",
            owner_user_id="user-1",
        )

        results = await graph_store._find_memories_about_persons(
            person_ids={person.id},
        )

        assert len(results) == 1
        assert results[0].id == m1.id

    async def test_excludes_owner(self, graph_store: GraphStore):
        """Excludes memories owned by the specified user."""
        person = await graph_store.create_person(created_by="user-1", name="Bob")

        await graph_store._store.add_memory(
            content="Bob likes hiking",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        m2 = await graph_store._store.add_memory(
            content="Bob enjoys music",
            owner_user_id="user-2",
            subject_person_ids=[person.id],
        )

        results = await graph_store._find_memories_about_persons(
            person_ids={person.id},
            exclude_owner_user_id="user-1",
        )

        assert len(results) == 1
        assert results[0].id == m2.id

    async def test_excludes_archived(self, graph_store: GraphStore):
        """Archived memories are not returned."""
        person = await graph_store.create_person(created_by="user-1", name="Bob")

        m1 = await graph_store._store.add_memory(
            content="Bob likes hiking",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        # Archive it
        await graph_store._store.delete_memory(m1.id)

        results = await graph_store._find_memories_about_persons(
            person_ids={person.id},
        )

        assert len(results) == 0

    async def test_excludes_superseded(self, graph_store: GraphStore):
        """Superseded memories are not returned."""
        person = await graph_store.create_person(created_by="user-1", name="Bob")

        old_m = await graph_store._store.add_memory(
            content="Bob likes red",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        new_m = await graph_store._store.add_memory(
            content="Bob likes blue",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        await graph_store._store.mark_memory_superseded(old_m.id, new_m.id)

        # Rebuild graph after supersession
        graph_store._graph_built = False

        results = await graph_store._find_memories_about_persons(
            person_ids={person.id},
        )

        assert len(results) == 1
        assert results[0].id == new_m.id

    async def test_excludes_non_portable(self, graph_store: GraphStore):
        """Non-portable memories are excluded when portable_only=True."""
        person = await graph_store.create_person(created_by="user-1", name="Bob")

        await graph_store._store.add_memory(
            content="Bob is presenting next",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
            portable=False,
        )
        m2 = await graph_store._store.add_memory(
            content="Bob likes hiking",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
            portable=True,
        )

        results = await graph_store._find_memories_about_persons(
            person_ids={person.id},
            portable_only=True,
        )

        assert len(results) == 1
        assert results[0].id == m2.id

    async def test_multiple_person_ids(self, graph_store: GraphStore):
        """Finds memories about any of the given person IDs."""
        p1 = await graph_store.create_person(created_by="user-1", name="Bob")
        p2 = await graph_store.create_person(created_by="user-1", name="Carol")

        m1 = await graph_store._store.add_memory(
            content="Bob likes hiking",
            owner_user_id="user-1",
            subject_person_ids=[p1.id],
        )
        m2 = await graph_store._store.add_memory(
            content="Carol likes cooking",
            owner_user_id="user-1",
            subject_person_ids=[p2.id],
        )

        results = await graph_store._find_memories_about_persons(
            person_ids={p1.id, p2.id},
        )

        result_ids = {r.id for r in results}
        assert result_ids == {m1.id, m2.id}


class TestForgetPersonViaGraph:
    """Test that forget_person uses graph index."""

    async def test_archives_memories_about_person(self, graph_store: GraphStore):
        """forget_person archives memories with ABOUT edges to the person."""
        person = await graph_store.create_person(created_by="user-1", name="Bob")

        about_bob = await graph_store._store.add_memory(
            content="Bob likes hiking",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        unrelated = await graph_store._store.add_memory(
            content="Alice likes cooking",
            owner_user_id="user-1",
        )

        archived_count = await graph_store.forget_person(person_id=person.id)

        assert archived_count == 1
        assert await graph_store._store.get_memory(about_bob.id) is None
        assert await graph_store._store.get_memory(unrelated.id) is not None

    async def test_returns_zero_for_unknown_person(self, graph_store: GraphStore):
        """Returns 0 when person has no memories."""
        archived_count = await graph_store.forget_person(person_id="nonexistent")
        assert archived_count == 0


class TestFindPersonIdsForUsernameViaGraph:
    """Test graph-based username -> person resolution."""

    async def test_resolves_via_user_node(self, graph_store: GraphStore):
        """Resolves username -> user node -> person via IS_PERSON edge."""
        person = await graph_store.create_person(
            created_by="100",
            name="Alice",
            aliases=["alice"],
            relationship="self",
            relationship_stated_by="100",
        )

        # Create a user node linked to this person
        await graph_store.ensure_user(
            provider="telegram",
            provider_id="100",
            username="alice",
            person_id=person.id,
        )

        result = await graph_store.find_person_ids_for_username("alice")
        assert person.id in result

    async def test_falls_back_to_linear_scan(self, graph_store: GraphStore):
        """Falls back to linear scan when no user node exists."""
        person = await graph_store.create_person(
            created_by="system",
            name="alice",
        )

        # No user node created — should fall back to linear scan
        result = await graph_store.find_person_ids_for_username("alice")
        assert person.id in result

    async def test_strips_at_prefix(self, graph_store: GraphStore):
        """Handles @username prefix."""
        person = await graph_store.create_person(
            created_by="100",
            name="Bob",
            aliases=["bob"],
        )

        await graph_store.ensure_user(
            provider="telegram",
            provider_id="100",
            username="bob",
            person_id=person.id,
        )

        result = await graph_store.find_person_ids_for_username("@bob")
        assert person.id in result


class TestGetContextViaGraph:
    """Test that get_context_for_message uses graph for cross-context retrieval."""

    async def test_cross_context_retrieval_via_graph(self, graph_store: GraphStore):
        """Cross-context memories are found via graph lookups."""
        person = await graph_store.create_person(created_by="user-1", name="Bob")

        # Memory about Bob from user-2 (cross-context)
        cross_mem = await graph_store._store.add_memory(
            content="Bob likes hiking",
            owner_user_id="user-2",
            subject_person_ids=[person.id],
        )

        # Wire mock index to return no direct results
        graph_store._index.search = AsyncMock(return_value=[])  # type: ignore[assignment]

        ctx = await graph_store.get_context_for_message(
            user_id="user-1",
            user_message="tell me about Bob",
            participant_person_ids={"bob": {person.id}},
        )

        # Should find the cross-context memory
        memory_ids = {m.id for m in ctx.memories}
        assert cross_mem.id in memory_ids

    async def test_cross_context_privacy_filter(self, graph_store: GraphStore):
        """SENSITIVE memories about others are filtered in group chats."""
        person = await graph_store.create_person(created_by="user-2", name="Bob")

        # SENSITIVE memory about Bob from user-2
        await graph_store._store.add_memory(
            content="Bob has health issue",
            owner_user_id="user-2",
            subject_person_ids=[person.id],
            sensitivity=Sensitivity.SENSITIVE,
        )

        graph_store._index.search = AsyncMock(return_value=[])  # type: ignore[assignment]

        ctx = await graph_store.get_context_for_message(
            user_id="user-1",
            user_message="tell me about Bob",
            chat_type="group",
            participant_person_ids={"bob": {person.id}},
        )

        # SENSITIVE memory about Bob should NOT appear (querier is not subject)
        assert len(ctx.memories) == 0
