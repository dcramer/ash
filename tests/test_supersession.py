"""Tests for supersession correctness paths.

Tests _is_protected_by_subject_authority, supersede_confirmed_hearsay,
and the LLM verification branch (similarity 0.75-0.85).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.graph.graph import KnowledgeGraph
from ash.graph.persistence import GraphPersistence
from ash.memory.embeddings import EmbeddingGenerator
from ash.store.store import Store
from ash.store.types import MemoryEntry, MemoryType


def _make_search_results(
    results: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    return results


@pytest.fixture
def mock_index():
    index = MagicMock()
    index.search = MagicMock(return_value=[])
    index.add = MagicMock()
    index.remove = MagicMock()
    index.save = AsyncMock()
    index.get_ids = MagicMock(return_value=set())
    return index


@pytest.fixture
def mock_embedding_generator():
    generator = MagicMock(spec=EmbeddingGenerator)
    generator.embed = AsyncMock(return_value=[0.1] * 1536)
    return generator


@pytest.fixture
async def graph_store(graph_dir, mock_index, mock_embedding_generator) -> Store:
    graph = KnowledgeGraph()
    persistence = GraphPersistence(graph_dir)
    store = Store(
        graph=graph,
        persistence=persistence,
        vector_index=mock_index,
        embedding_generator=mock_embedding_generator,
    )
    store._llm_model = "mock-model"
    return store


class TestIsProtectedBySubjectAuthority:
    """Tests for _is_protected_by_subject_authority.

    This method prevents third-party hearsay from overwriting
    first-person facts (e.g., Alice's gossip about Bob shouldn't
    overwrite what Bob said about himself).
    """

    def _add_memory_with_subjects(
        self,
        graph_store: Store,
        memory: MemoryEntry,
        subject_person_ids: list[str],
    ) -> None:
        """Add a memory to the graph and create ABOUT edges."""
        from ash.graph.edges import create_about_edge

        graph_store.graph.add_memory(memory)
        for pid in subject_person_ids:
            graph_store.graph.add_edge(create_about_edge(memory.id, pid))

    async def test_not_protected_when_same_source(self, graph_store: Store):
        """If both memories come from the same source, no protection."""
        person = await graph_store.create_person(
            created_by="bob", name="Bob", aliases=["bob"]
        )
        await graph_store.ensure_user(
            provider="test", provider_id="bob", username="bob", person_id=person.id
        )

        candidate = MemoryEntry(
            id="old-1",
            content="Bob likes blue",
            source_username="bob",
            memory_type=MemoryType.KNOWLEDGE,
        )
        new_memory = MemoryEntry(
            id="new-1",
            content="Bob likes red",
            source_username="bob",
            memory_type=MemoryType.KNOWLEDGE,
        )
        self._add_memory_with_subjects(graph_store, candidate, [person.id])
        self._add_memory_with_subjects(graph_store, new_memory, [person.id])

        result = await graph_store._is_protected_by_subject_authority(
            candidate, new_memory
        )
        assert result is False

    async def test_not_protected_when_subject_speaks_about_self(
        self, graph_store: Store
    ):
        """When the new memory's source IS the subject, not protected."""
        person = await graph_store.create_person(
            created_by="bob", name="Bob", aliases=["bob"]
        )
        await graph_store.ensure_user(
            provider="test", provider_id="bob", username="bob", person_id=person.id
        )
        await graph_store.ensure_user(
            provider="test", provider_id="alice", username="alice"
        )

        # Third party wrote original
        candidate = MemoryEntry(
            id="old-1",
            content="Bob's favorite color is blue",
            source_username="alice",
            memory_type=MemoryType.KNOWLEDGE,
        )
        # Bob himself provides the update
        new_memory = MemoryEntry(
            id="new-1",
            content="Bob's favorite color is red",
            source_username="bob",
            memory_type=MemoryType.KNOWLEDGE,
        )
        self._add_memory_with_subjects(graph_store, candidate, [person.id])
        self._add_memory_with_subjects(graph_store, new_memory, [person.id])

        result = await graph_store._is_protected_by_subject_authority(
            candidate, new_memory
        )
        assert result is False

    async def test_protected_when_subject_authored_and_third_party_updates(
        self, graph_store: Store
    ):
        """First-person fact is protected from third-party overwrite."""
        bob = await graph_store.create_person(
            created_by="bob", name="Bob", aliases=["bob"]
        )
        await graph_store.ensure_user(
            provider="test", provider_id="bob", username="bob", person_id=bob.id
        )
        await graph_store.ensure_user(
            provider="test", provider_id="alice", username="alice"
        )

        # Bob said it about himself
        candidate = MemoryEntry(
            id="old-1",
            content="I like blue",
            source_username="bob",
            memory_type=MemoryType.KNOWLEDGE,
        )
        # Alice tries to overwrite
        new_memory = MemoryEntry(
            id="new-1",
            content="Bob likes red",
            source_username="alice",
            memory_type=MemoryType.KNOWLEDGE,
        )
        self._add_memory_with_subjects(graph_store, candidate, [bob.id])
        self._add_memory_with_subjects(graph_store, new_memory, [bob.id])

        result = await graph_store._is_protected_by_subject_authority(
            candidate, new_memory
        )
        assert result is True

    async def test_not_protected_when_no_source_username(self, graph_store: Store):
        """If either memory lacks source_username, not protected."""
        person = await graph_store.create_person(created_by="user", name="Bob")

        candidate = MemoryEntry(
            id="old-1",
            content="Bob likes blue",
            source_username=None,
            memory_type=MemoryType.KNOWLEDGE,
        )
        new_memory = MemoryEntry(
            id="new-1",
            content="Bob likes red",
            source_username="alice",
            memory_type=MemoryType.KNOWLEDGE,
        )
        self._add_memory_with_subjects(graph_store, candidate, [person.id])
        self._add_memory_with_subjects(graph_store, new_memory, [person.id])

        result = await graph_store._is_protected_by_subject_authority(
            candidate, new_memory
        )
        assert result is False

    async def test_not_protected_when_no_subject_person_ids(self, graph_store: Store):
        """If candidate has no subjects (no ABOUT edges), not protected."""
        candidate = MemoryEntry(
            id="old-1",
            content="The weather is nice",
            source_username="bob",
            memory_type=MemoryType.KNOWLEDGE,
        )
        new_memory = MemoryEntry(
            id="new-1",
            content="The weather is bad",
            source_username="alice",
            memory_type=MemoryType.KNOWLEDGE,
        )
        # Add to graph with no ABOUT edges
        graph_store.graph.add_memory(candidate)
        graph_store.graph.add_memory(new_memory)

        result = await graph_store._is_protected_by_subject_authority(
            candidate, new_memory
        )
        assert result is False

    async def test_exception_returns_true_protected(self, graph_store: Store):
        """On exception, returns True (fail-safe: protects the memory)."""
        person = await graph_store.create_person(
            created_by="bob", name="Bob", aliases=["bob"]
        )

        candidate = MemoryEntry(
            id="old-1",
            content="Bob likes blue",
            source_username="bob",
            memory_type=MemoryType.KNOWLEDGE,
        )
        new_memory = MemoryEntry(
            id="new-1",
            content="Bob likes red",
            source_username="alice",
            memory_type=MemoryType.KNOWLEDGE,
        )
        self._add_memory_with_subjects(graph_store, candidate, [person.id])
        self._add_memory_with_subjects(graph_store, new_memory, [person.id])

        # Force an exception in find_person_ids_for_username
        original = graph_store.find_person_ids_for_username
        graph_store.find_person_ids_for_username = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("DB connection lost")
        )

        result = await graph_store._is_protected_by_subject_authority(
            candidate, new_memory
        )
        assert result is True

        graph_store.find_person_ids_for_username = original  # type: ignore[method-assign]


class TestSupersedeConflictingMemories:
    """Tests for the LLM verification branch in supersession."""

    async def test_high_similarity_skips_llm(self, graph_store: Store):
        """Similarity >= 0.85 should supersede without LLM call."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock()
        graph_store._llm = mock_llm

        m_old = await graph_store.add_memory(
            content="User likes red",
            owner_user_id="user-1",
        )
        m_new = await graph_store.add_memory(
            content="User likes blue",
            owner_user_id="user-1",
        )

        # Mock index to return high similarity
        graph_store._index.search = MagicMock(  # type: ignore[assignment]
            return_value=_make_search_results([(m_old.id, 0.90)])
        )

        count = await graph_store.supersede_conflicting_memories(
            new_memory=m_new,
            owner_user_id="user-1",
        )

        assert count == 1
        mock_llm.complete.assert_not_called()

    async def test_medium_similarity_triggers_llm(self, graph_store: Store):
        """Similarity between 0.75 and 0.85 should trigger LLM verification."""
        from ash.llm.types import Message, Role

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=MagicMock(message=Message(role=Role.ASSISTANT, content="YES"))
        )
        graph_store._llm = mock_llm

        m_old = await graph_store.add_memory(
            content="User's birthday is in May",
            owner_user_id="user-1",
        )
        m_new = await graph_store.add_memory(
            content="User's birthday is in June",
            owner_user_id="user-1",
        )

        graph_store._index.search = MagicMock(  # type: ignore[assignment]
            return_value=_make_search_results([(m_old.id, 0.80)])
        )

        count = await graph_store.supersede_conflicting_memories(
            new_memory=m_new,
            owner_user_id="user-1",
        )

        assert count == 1
        mock_llm.complete.assert_called_once()

    async def test_medium_similarity_llm_says_no(self, graph_store: Store):
        """When LLM says NO, memory should not be superseded."""
        from ash.llm.types import Message, Role

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=MagicMock(message=Message(role=Role.ASSISTANT, content="NO"))
        )
        graph_store._llm = mock_llm

        m_old = await graph_store.add_memory(
            content="User likes hiking",
            owner_user_id="user-1",
        )
        m_new = await graph_store.add_memory(
            content="User likes swimming",
            owner_user_id="user-1",
        )

        graph_store._index.search = MagicMock(  # type: ignore[assignment]
            return_value=_make_search_results([(m_old.id, 0.80)])
        )

        count = await graph_store.supersede_conflicting_memories(
            new_memory=m_new,
            owner_user_id="user-1",
        )

        assert count == 0


class TestBatchMarkSupersededGuards:
    """Tests for defensive guards in batch_mark_superseded."""

    async def test_skips_self_supersession_pair(self, graph_store: Store):
        memory = await graph_store.add_memory(
            content="User likes espresso",
            owner_user_id="user-1",
        )

        marked = await graph_store.batch_mark_superseded([(memory.id, memory.id)])

        assert marked == []
        assert graph_store.graph.memories[memory.id].superseded_at is None

    async def test_skips_pair_that_would_create_cycle(self, graph_store: Store):
        from ash.graph.edges import create_supersedes_edge

        old = await graph_store.add_memory(
            content="User uses vim",
            owner_user_id="user-1",
        )
        new = await graph_store.add_memory(
            content="User switched to neovim",
            owner_user_id="user-1",
        )

        # Existing edge old -> new means adding new -> old would create a cycle.
        graph_store.graph.add_edge(create_supersedes_edge(old.id, new.id))

        marked = await graph_store.batch_mark_superseded([(old.id, new.id)])

        assert marked == []
        assert graph_store.graph.memories[old.id].superseded_at is None


class TestSupersedeConfirmedHearsay:
    """Tests for supersede_confirmed_hearsay."""

    async def test_hearsay_superseded_by_first_person(self, graph_store: Store):
        """Third-party hearsay is superseded when the subject speaks."""
        bob = await graph_store.create_person(
            created_by="bob", name="Bob", aliases=["bob"]
        )

        # Hearsay: Alice says Bob likes hiking
        hearsay = await graph_store.add_memory(
            content="Bob likes hiking",
            owner_user_id="user-1",
            source_username="alice",
            subject_person_ids=[bob.id],
        )

        # First-person: Bob says he likes swimming
        fact = await graph_store.add_memory(
            content="I like swimming",
            owner_user_id="user-1",
            source_username="bob",
            subject_person_ids=[bob.id],
        )

        # Mock index: searching fact content returns the hearsay as similar
        graph_store._index.search = MagicMock(  # type: ignore[assignment]
            return_value=_make_search_results([(hearsay.id, 0.88)])
        )

        count = await graph_store.supersede_confirmed_hearsay(
            new_memory=fact,
            person_ids={bob.id},
            source_username="bob",
        )

        assert count == 1
        # Verify hearsay was actually marked superseded
        updated = await graph_store.get_memory(hearsay.id)
        assert updated is not None
        assert updated.superseded_at is not None

    async def test_first_person_not_superseded_by_hearsay(self, graph_store: Store):
        """First-person facts should not be treated as hearsay candidates."""
        bob = await graph_store.create_person(
            created_by="bob", name="Bob", aliases=["bob"]
        )

        # First-person fact: Bob's own statement
        fact = await graph_store.add_memory(
            content="I like hiking",
            owner_user_id="user-1",
            source_username="bob",
            subject_person_ids=[bob.id],
        )

        # Another statement from Bob
        new_fact = await graph_store.add_memory(
            content="I like swimming",
            owner_user_id="user-1",
            source_username="bob",
            subject_person_ids=[bob.id],
        )

        graph_store._index.search = MagicMock(  # type: ignore[assignment]
            return_value=_make_search_results([(new_fact.id, 0.90)])
        )

        # This should NOT supersede the original fact (both are from bob)
        count = await graph_store.supersede_confirmed_hearsay(
            new_memory=new_fact,
            person_ids={bob.id},
            source_username="bob",
        )

        assert count == 0
        # Original fact should still be active
        original = await graph_store.get_memory(fact.id)
        assert original is not None
        assert original.superseded_at is None

    async def test_below_threshold_not_superseded(self, graph_store: Store):
        """Hearsay below similarity threshold is not superseded."""
        bob = await graph_store.create_person(
            created_by="bob", name="Bob", aliases=["bob"]
        )

        await graph_store.add_memory(
            content="Bob works at Google",
            owner_user_id="user-1",
            source_username="alice",
            subject_person_ids=[bob.id],
        )

        fact = await graph_store.add_memory(
            content="I like swimming",
            owner_user_id="user-1",
            source_username="bob",
            subject_person_ids=[bob.id],
        )

        # Low similarity — different topics
        graph_store._index.search = MagicMock(  # type: ignore[assignment]
            return_value=_make_search_results([(fact.id, 0.40)])
        )

        count = await graph_store.supersede_confirmed_hearsay(
            new_memory=fact,
            person_ids={bob.id},
            source_username="bob",
        )

        assert count == 0

    async def test_hearsay_similarity_exception_logged(self, graph_store: Store):
        """Exceptions checking individual hearsay are caught and logged."""
        bob = await graph_store.create_person(
            created_by="bob", name="Bob", aliases=["bob"]
        )

        await graph_store.add_memory(
            content="Bob likes hiking",
            owner_user_id="user-1",
            source_username="alice",
            subject_person_ids=[bob.id],
        )

        fact = await graph_store.add_memory(
            content="I like swimming",
            owner_user_id="user-1",
            source_username="bob",
            subject_person_ids=[bob.id],
        )

        # Simulate search failure
        graph_store._index.search = MagicMock(  # type: ignore[assignment]
            side_effect=RuntimeError("Search failed")
        )

        # Should not raise — just logs warning and returns 0
        count = await graph_store.supersede_confirmed_hearsay(
            new_memory=fact,
            person_ids={bob.id},
            source_username="bob",
        )

        assert count == 0
