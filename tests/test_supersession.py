"""Tests for supersession correctness paths.

Tests _is_protected_by_subject_authority, supersede_confirmed_hearsay,
and the LLM verification branch (similarity 0.75-0.85).
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.graph.store import GraphStore
from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.file_store import FileMemoryStore
from ash.memory.index import VectorIndex, VectorSearchResult
from ash.memory.types import MemoryEntry, MemoryType


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


class TestIsProtectedBySubjectAuthority:
    """Tests for _is_protected_by_subject_authority.

    This method prevents third-party hearsay from overwriting
    first-person facts (e.g., Alice's gossip about Bob shouldn't
    overwrite what Bob said about himself).
    """

    async def test_not_protected_when_same_source(self, graph_store: GraphStore):
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
            subject_person_ids=[person.id],
            memory_type=MemoryType.KNOWLEDGE,
        )
        new_memory = MemoryEntry(
            id="new-1",
            content="Bob likes red",
            source_username="bob",
            subject_person_ids=[person.id],
            memory_type=MemoryType.KNOWLEDGE,
        )

        result = await graph_store._is_protected_by_subject_authority(
            candidate, new_memory
        )
        assert result is False

    async def test_not_protected_when_subject_speaks_about_self(
        self, graph_store: GraphStore
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
            subject_person_ids=[person.id],
            memory_type=MemoryType.KNOWLEDGE,
        )
        # Bob himself provides the update
        new_memory = MemoryEntry(
            id="new-1",
            content="Bob's favorite color is red",
            source_username="bob",
            subject_person_ids=[person.id],
            memory_type=MemoryType.KNOWLEDGE,
        )

        result = await graph_store._is_protected_by_subject_authority(
            candidate, new_memory
        )
        assert result is False

    async def test_protected_when_subject_authored_and_third_party_updates(
        self, graph_store: GraphStore
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
            subject_person_ids=[bob.id],
            memory_type=MemoryType.KNOWLEDGE,
        )
        # Alice tries to overwrite
        new_memory = MemoryEntry(
            id="new-1",
            content="Bob likes red",
            source_username="alice",
            subject_person_ids=[bob.id],
            memory_type=MemoryType.KNOWLEDGE,
        )

        result = await graph_store._is_protected_by_subject_authority(
            candidate, new_memory
        )
        assert result is True

    async def test_not_protected_when_no_source_username(self, graph_store: GraphStore):
        """If either memory lacks source_username, not protected."""
        person = await graph_store.create_person(created_by="user", name="Bob")

        candidate = MemoryEntry(
            id="old-1",
            content="Bob likes blue",
            source_username=None,
            subject_person_ids=[person.id],
            memory_type=MemoryType.KNOWLEDGE,
        )
        new_memory = MemoryEntry(
            id="new-1",
            content="Bob likes red",
            source_username="alice",
            subject_person_ids=[person.id],
            memory_type=MemoryType.KNOWLEDGE,
        )

        result = await graph_store._is_protected_by_subject_authority(
            candidate, new_memory
        )
        assert result is False

    async def test_not_protected_when_no_subject_person_ids(
        self, graph_store: GraphStore
    ):
        """If candidate has no subject_person_ids, not protected."""
        candidate = MemoryEntry(
            id="old-1",
            content="The weather is nice",
            source_username="bob",
            subject_person_ids=[],
            memory_type=MemoryType.KNOWLEDGE,
        )
        new_memory = MemoryEntry(
            id="new-1",
            content="The weather is bad",
            source_username="alice",
            memory_type=MemoryType.KNOWLEDGE,
        )

        result = await graph_store._is_protected_by_subject_authority(
            candidate, new_memory
        )
        assert result is False

    async def test_exception_returns_true_protected(self, graph_store: GraphStore):
        """On exception, returns True (fail-safe: protects the memory)."""
        person = await graph_store.create_person(
            created_by="bob", name="Bob", aliases=["bob"]
        )

        candidate = MemoryEntry(
            id="old-1",
            content="Bob likes blue",
            source_username="bob",
            subject_person_ids=[person.id],
            memory_type=MemoryType.KNOWLEDGE,
        )
        new_memory = MemoryEntry(
            id="new-1",
            content="Bob likes red",
            source_username="alice",
            subject_person_ids=[person.id],
            memory_type=MemoryType.KNOWLEDGE,
        )

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

    async def test_high_similarity_skips_llm(self, graph_store: GraphStore):
        """Similarity >= 0.85 should supersede without LLM call."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock()
        graph_store._llm = mock_llm

        m_old = await graph_store._store.add_memory(
            content="User likes red",
            owner_user_id="user-1",
        )
        m_new = await graph_store._store.add_memory(
            content="User likes blue",
            owner_user_id="user-1",
        )

        # Mock index to return high similarity
        graph_store._index.search = AsyncMock(  # type: ignore[assignment]
            return_value=_make_search_results([(m_old.id, 0.90)])
        )

        count = await graph_store.supersede_conflicting_memories(
            new_memory=m_new,
            owner_user_id="user-1",
        )

        assert count == 1
        mock_llm.complete.assert_not_called()

    async def test_medium_similarity_triggers_llm(self, graph_store: GraphStore):
        """Similarity between 0.75 and 0.85 should trigger LLM verification."""
        from ash.llm.types import Message, Role

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=MagicMock(message=Message(role=Role.ASSISTANT, content="YES"))
        )
        graph_store._llm = mock_llm

        m_old = await graph_store._store.add_memory(
            content="User's birthday is in May",
            owner_user_id="user-1",
        )
        m_new = await graph_store._store.add_memory(
            content="User's birthday is in June",
            owner_user_id="user-1",
        )

        graph_store._index.search = AsyncMock(  # type: ignore[assignment]
            return_value=_make_search_results([(m_old.id, 0.80)])
        )

        count = await graph_store.supersede_conflicting_memories(
            new_memory=m_new,
            owner_user_id="user-1",
        )

        assert count == 1
        mock_llm.complete.assert_called_once()

    async def test_medium_similarity_llm_says_no(self, graph_store: GraphStore):
        """When LLM says NO, memory should not be superseded."""
        from ash.llm.types import Message, Role

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=MagicMock(message=Message(role=Role.ASSISTANT, content="NO"))
        )
        graph_store._llm = mock_llm

        m_old = await graph_store._store.add_memory(
            content="User likes hiking",
            owner_user_id="user-1",
        )
        m_new = await graph_store._store.add_memory(
            content="User likes swimming",
            owner_user_id="user-1",
        )

        graph_store._index.search = AsyncMock(  # type: ignore[assignment]
            return_value=_make_search_results([(m_old.id, 0.80)])
        )

        count = await graph_store.supersede_conflicting_memories(
            new_memory=m_new,
            owner_user_id="user-1",
        )

        assert count == 0


class TestSupersedeConfirmedHearsay:
    """Tests for supersede_confirmed_hearsay."""

    async def test_hearsay_superseded_by_first_person(self, graph_store: GraphStore):
        """Third-party hearsay is superseded when the subject speaks."""
        bob = await graph_store.create_person(
            created_by="bob", name="Bob", aliases=["bob"]
        )

        # Hearsay: Alice says Bob likes hiking
        hearsay = await graph_store._store.add_memory(
            content="Bob likes hiking",
            owner_user_id="user-1",
            source_username="alice",
            subject_person_ids=[bob.id],
        )

        # First-person: Bob says he likes swimming
        fact = await graph_store._store.add_memory(
            content="I like swimming",
            owner_user_id="user-1",
            source_username="bob",
            subject_person_ids=[bob.id],
        )

        # Mock index: when searching hearsay content, return the fact
        graph_store._index.search = AsyncMock(  # type: ignore[assignment]
            return_value=_make_search_results([(fact.id, 0.88)])
        )

        count = await graph_store.supersede_confirmed_hearsay(
            new_memory=fact,
            person_ids={bob.id},
            source_username="bob",
            owner_user_id="user-1",
        )

        assert count == 1
        # Verify hearsay was actually marked superseded
        updated = await graph_store._store.get_memory(hearsay.id)
        assert updated is not None
        assert updated.superseded_at is not None

    async def test_first_person_not_superseded_by_hearsay(
        self, graph_store: GraphStore
    ):
        """First-person facts should not be treated as hearsay candidates."""
        bob = await graph_store.create_person(
            created_by="bob", name="Bob", aliases=["bob"]
        )

        # First-person fact: Bob's own statement
        fact = await graph_store._store.add_memory(
            content="I like hiking",
            owner_user_id="user-1",
            source_username="bob",
            subject_person_ids=[bob.id],
        )

        # Another statement from Bob
        new_fact = await graph_store._store.add_memory(
            content="I like swimming",
            owner_user_id="user-1",
            source_username="bob",
            subject_person_ids=[bob.id],
        )

        graph_store._index.search = AsyncMock(  # type: ignore[assignment]
            return_value=_make_search_results([(new_fact.id, 0.90)])
        )

        # This should NOT supersede the original fact (both are from bob)
        count = await graph_store.supersede_confirmed_hearsay(
            new_memory=new_fact,
            person_ids={bob.id},
            source_username="bob",
            owner_user_id="user-1",
        )

        assert count == 0
        # Original fact should still be active
        original = await graph_store._store.get_memory(fact.id)
        assert original is not None
        assert original.superseded_at is None

    async def test_below_threshold_not_superseded(self, graph_store: GraphStore):
        """Hearsay below similarity threshold is not superseded."""
        bob = await graph_store.create_person(
            created_by="bob", name="Bob", aliases=["bob"]
        )

        await graph_store._store.add_memory(
            content="Bob works at Google",
            owner_user_id="user-1",
            source_username="alice",
            subject_person_ids=[bob.id],
        )

        fact = await graph_store._store.add_memory(
            content="I like swimming",
            owner_user_id="user-1",
            source_username="bob",
            subject_person_ids=[bob.id],
        )

        # Low similarity — different topics
        graph_store._index.search = AsyncMock(  # type: ignore[assignment]
            return_value=_make_search_results([(fact.id, 0.40)])
        )

        count = await graph_store.supersede_confirmed_hearsay(
            new_memory=fact,
            person_ids={bob.id},
            source_username="bob",
            owner_user_id="user-1",
        )

        assert count == 0

    async def test_hearsay_similarity_exception_logged(self, graph_store: GraphStore):
        """Exceptions checking individual hearsay are caught and logged."""
        bob = await graph_store.create_person(
            created_by="bob", name="Bob", aliases=["bob"]
        )

        await graph_store._store.add_memory(
            content="Bob likes hiking",
            owner_user_id="user-1",
            source_username="alice",
            subject_person_ids=[bob.id],
        )

        fact = await graph_store._store.add_memory(
            content="I like swimming",
            owner_user_id="user-1",
            source_username="bob",
            subject_person_ids=[bob.id],
        )

        # Simulate search failure
        graph_store._index.search = AsyncMock(  # type: ignore[assignment]
            side_effect=RuntimeError("Search failed")
        )

        # Should not raise — just logs warning and returns 0
        count = await graph_store.supersede_confirmed_hearsay(
            new_memory=fact,
            person_ids={bob.id},
            source_username="bob",
            owner_user_id="user-1",
        )

        assert count == 0
