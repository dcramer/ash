"""Tests for Store graph-based optimizations.

Verifies that Store uses SQL-indexed lookups for O(1) queries
instead of linear scans over all memories.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.db.engine import Database
from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.index import VectorIndex, VectorSearchResult
from ash.store.store import Store
from ash.store.types import Sensitivity


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
    index.delete_embeddings = AsyncMock()
    return index


@pytest.fixture
def mock_embedding_generator():
    generator = MagicMock(spec=EmbeddingGenerator)
    generator.embed = AsyncMock(return_value=[0.1] * 1536)
    return generator


@pytest.fixture
async def graph_store(
    database: Database, mock_index, mock_embedding_generator
) -> Store:
    return Store(
        db=database,
        vector_index=mock_index,
        embedding_generator=mock_embedding_generator,
    )


class TestFindMemoriesAboutPersons:
    """Test graph-based memory lookup by subject person."""

    async def test_finds_memories_about_person(self, graph_store: Store):
        """Graph-based lookup returns memories about a given person."""
        person = await graph_store.create_person(created_by="user-1", name="Bob")

        m1 = await graph_store.add_memory(
            content="Bob likes hiking",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        await graph_store.add_memory(
            content="Alice likes cooking",
            owner_user_id="user-1",
        )

        results = await graph_store._find_memories_about_persons(
            person_ids={person.id},
        )

        assert len(results) == 1
        assert results[0].id == m1.id

    async def test_excludes_owner(self, graph_store: Store):
        """Excludes memories owned by the specified user."""
        person = await graph_store.create_person(created_by="user-1", name="Bob")

        await graph_store.add_memory(
            content="Bob likes hiking",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        m2 = await graph_store.add_memory(
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

    async def test_excludes_archived(self, graph_store: Store):
        """Archived memories are not returned."""
        person = await graph_store.create_person(created_by="user-1", name="Bob")

        m1 = await graph_store.add_memory(
            content="Bob likes hiking",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        # Archive it
        await graph_store.delete_memory(m1.id)

        results = await graph_store._find_memories_about_persons(
            person_ids={person.id},
        )

        assert len(results) == 0

    async def test_excludes_superseded(self, graph_store: Store):
        """Superseded memories are not returned."""
        person = await graph_store.create_person(created_by="user-1", name="Bob")

        old_m = await graph_store.add_memory(
            content="Bob likes red",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        new_m = await graph_store.add_memory(
            content="Bob likes blue",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        await graph_store.batch_mark_superseded([(old_m.id, new_m.id)])

        results = await graph_store._find_memories_about_persons(
            person_ids={person.id},
        )

        assert len(results) == 1
        assert results[0].id == new_m.id

    async def test_excludes_non_portable(self, graph_store: Store):
        """Non-portable memories are excluded when portable_only=True."""
        person = await graph_store.create_person(created_by="user-1", name="Bob")

        await graph_store.add_memory(
            content="Bob is presenting next",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
            portable=False,
        )
        m2 = await graph_store.add_memory(
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

    async def test_multiple_person_ids(self, graph_store: Store):
        """Finds memories about any of the given person IDs."""
        p1 = await graph_store.create_person(created_by="user-1", name="Bob")
        p2 = await graph_store.create_person(created_by="user-1", name="Carol")

        m1 = await graph_store.add_memory(
            content="Bob likes hiking",
            owner_user_id="user-1",
            subject_person_ids=[p1.id],
        )
        m2 = await graph_store.add_memory(
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

    async def test_archives_memories_about_person(self, graph_store: Store):
        """forget_person archives memories with ABOUT edges to the person."""
        person = await graph_store.create_person(created_by="user-1", name="Bob")

        about_bob = await graph_store.add_memory(
            content="Bob likes hiking",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        unrelated = await graph_store.add_memory(
            content="Alice likes cooking",
            owner_user_id="user-1",
        )

        archived_count = await graph_store.forget_person(person_id=person.id)

        assert archived_count == 1
        assert await graph_store.get_memory(about_bob.id) is None
        assert await graph_store.get_memory(unrelated.id) is not None

    async def test_returns_zero_for_unknown_person(self, graph_store: Store):
        """Returns 0 when person has no memories."""
        archived_count = await graph_store.forget_person(person_id="nonexistent")
        assert archived_count == 0


class TestFindPersonIdsForUsernameViaGraph:
    """Test graph-based username -> person resolution."""

    async def test_resolves_via_user_node(self, graph_store: Store):
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

    async def test_falls_back_to_linear_scan(self, graph_store: Store):
        """Falls back to linear scan when no user node exists."""
        person = await graph_store.create_person(
            created_by="system",
            name="alice",
        )

        # No user node created — should fall back to linear scan
        result = await graph_store.find_person_ids_for_username("alice")
        assert person.id in result

    async def test_strips_at_prefix(self, graph_store: Store):
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

    async def test_cross_context_retrieval_via_graph(self, graph_store: Store):
        """Cross-context memories are found via graph lookups."""
        person = await graph_store.create_person(created_by="user-1", name="Bob")

        # Memory about Bob from user-2 (cross-context)
        cross_mem = await graph_store.add_memory(
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

    async def test_cross_context_privacy_filter(self, graph_store: Store):
        """SENSITIVE memories about others are filtered in group chats."""
        person = await graph_store.create_person(created_by="user-2", name="Bob")

        # SENSITIVE memory about Bob from user-2
        await graph_store.add_memory(
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


class TestGraphTraversalPass:
    """Tests for the graph traversal second pass in get_context_for_message.

    The graph traversal pass discovers additional memories about persons
    mentioned in the initial search results, even if those memories wouldn't
    match the query directly.
    """

    async def test_graph_traversal_finds_related_memories(self, graph_store: Store):
        """Memories about a mentioned person are surfaced via graph traversal."""
        alice = await graph_store.create_person(created_by="user-1", name="Alice")
        bob = await graph_store.create_person(created_by="user-1", name="Bob")

        # Memory about Alice that mentions Bob via subject_person_ids
        m_alice = await graph_store.add_memory(
            content="Alice and Bob went hiking together",
            owner_user_id="user-1",
            subject_person_ids=[alice.id, bob.id],
        )

        # Additional memory about Bob from user-2 (cross-context, not in search)
        m_bob_extra = await graph_store.add_memory(
            content="Bob is training for a marathon",
            owner_user_id="user-2",
            subject_person_ids=[bob.id],
        )

        # Mock index: direct search returns the Alice+Bob memory
        graph_store._index.search = AsyncMock(  # type: ignore[assignment]
            return_value=_make_search_results([(m_alice.id, 0.85)])
        )

        ctx = await graph_store.get_context_for_message(
            user_id="user-1",
            user_message="what's Alice been up to?",
        )

        memory_ids = {m.id for m in ctx.memories}
        # Should include both the direct hit AND the graph-traversal discovery
        assert m_alice.id in memory_ids
        assert m_bob_extra.id in memory_ids

    async def test_graph_traversal_excludes_already_seen(self, graph_store: Store):
        """Memories already in results are not duplicated by graph traversal."""
        person = await graph_store.create_person(created_by="user-1", name="Bob")

        m1 = await graph_store.add_memory(
            content="Bob likes hiking",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )

        # Same memory would show up in both search and graph traversal
        # but should only appear once
        graph_store._index.search = AsyncMock(  # type: ignore[assignment]
            return_value=_make_search_results([(m1.id, 0.90)])
        )

        ctx = await graph_store.get_context_for_message(
            user_id="user-1",
            user_message="tell me about Bob",
        )

        # Should not have duplicates
        ids = [m.id for m in ctx.memories]
        assert len(ids) == len(set(ids))

    async def test_graph_traversal_excludes_participant_persons(
        self, graph_store: Store
    ):
        """Graph traversal skips person IDs that are already handled as participants."""
        bob = await graph_store.create_person(created_by="user-1", name="Bob")
        carol = await graph_store.create_person(created_by="user-1", name="Carol")

        # Memory mentioning both Bob and Carol
        m1 = await graph_store.add_memory(
            content="Bob and Carol met at the park",
            owner_user_id="user-1",
            subject_person_ids=[bob.id, carol.id],
        )

        # Memory about Bob from user-2
        m_bob = await graph_store.add_memory(
            content="Bob enjoys running",
            owner_user_id="user-2",
            subject_person_ids=[bob.id],
        )

        graph_store._index.search = AsyncMock(  # type: ignore[assignment]
            return_value=_make_search_results([(m1.id, 0.85)])
        )

        # Bob is a participant — his memories are already handled in cross-context
        # Carol is NOT a participant — her graph traversal should work
        ctx = await graph_store.get_context_for_message(
            user_id="user-1",
            user_message="what happened at the park?",
            participant_person_ids={"bob": {bob.id}},
        )

        memory_ids = {m.id for m in ctx.memories}
        # Bob's extra memory should come from cross-context (pass 1), not graph traversal
        assert m_bob.id in memory_ids

    async def test_graph_traversal_marks_metadata(self, graph_store: Store):
        """Graph-traversal memories have graph_traversal=True in metadata."""
        person = await graph_store.create_person(created_by="user-1", name="Bob")

        m1 = await graph_store.add_memory(
            content="Bob was mentioned in a conversation",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )

        m_extra = await graph_store.add_memory(
            content="Bob's birthday is June 15",
            owner_user_id="user-2",
            subject_person_ids=[person.id],
        )

        graph_store._index.search = AsyncMock(  # type: ignore[assignment]
            return_value=_make_search_results([(m1.id, 0.85)])
        )

        ctx = await graph_store.get_context_for_message(
            user_id="user-1",
            user_message="any news about Bob?",
        )

        # Find the graph-traversal memory
        traversal_mems = [
            m for m in ctx.memories if (m.metadata or {}).get("graph_traversal")
        ]
        assert len(traversal_mems) >= 1
        assert traversal_mems[0].id == m_extra.id


class TestBatchMarkSuperseded:
    """Tests for Store.batch_mark_superseded."""

    async def test_batch_supersedes_and_deletes_embeddings(self, graph_store: Store):
        """batch_mark_superseded should mark memories and delete embeddings."""
        m1 = await graph_store.add_memory(content="Old 1")
        m2 = await graph_store.add_memory(content="Old 2")
        m3 = await graph_store.add_memory(content="New 1")
        m4 = await graph_store.add_memory(content="New 2")

        marked = await graph_store.batch_mark_superseded(
            [
                (m1.id, m3.id),
                (m2.id, m4.id),
            ]
        )

        assert set(marked) == {m1.id, m2.id}
        # Embeddings should be deleted for superseded memories
        graph_store._index.delete_embeddings.assert_called_once_with(marked)

    async def test_empty_pairs(self, graph_store: Store):
        """Empty pairs should be a no-op."""
        marked = await graph_store.batch_mark_superseded([])
        assert marked == []
        graph_store._index.delete_embeddings.assert_not_called()
