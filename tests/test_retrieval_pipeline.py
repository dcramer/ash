"""Tests for RetrievalPipeline.

Tests the multi-stage memory retrieval logic.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.graph.graph import KnowledgeGraph
from ash.graph.persistence import GraphPersistence
from ash.memory.embeddings import EmbeddingGenerator
from ash.store.retrieval import (
    RetrievalContext,
    RetrievalPipeline,
)
from ash.store.store import Store
from ash.store.types import SearchResult, Sensitivity


@pytest.fixture
def mock_store():
    """Create a mock Store for testing."""
    store = MagicMock()
    store.search = AsyncMock(return_value=[])
    store._resolve_subject_name = AsyncMock(return_value=None)
    return store


@pytest.fixture
def mock_embedding_generator():
    generator = MagicMock(spec=EmbeddingGenerator)
    generator.embed = AsyncMock(return_value=[0.1] * 1536)
    return generator


@pytest.fixture
def mock_index():
    index = MagicMock()
    index.search = MagicMock(return_value=[])
    index.add = MagicMock()
    index.remove = MagicMock()
    index.save = AsyncMock()
    return index


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


class TestRetrievalPipeline:
    """Tests for RetrievalPipeline stages."""

    @pytest.mark.asyncio
    async def test_primary_search_calls_store_search(self, mock_store):
        """Stage 1 should call store.search with correct parameters."""
        mock_store.search = AsyncMock(
            return_value=[
                SearchResult(
                    id="mem-1",
                    content="Test memory",
                    similarity=0.9,
                    metadata={},
                    source_type="memory",
                )
            ]
        )

        pipeline = RetrievalPipeline(mock_store)
        context = RetrievalContext(
            user_id="user-1",
            query="test query",
            chat_id="chat-1",
            max_memories=10,
        )

        result = await pipeline.retrieve(context)

        mock_store.search.assert_called_once_with(
            query="test query",
            limit=10,
            owner_user_id="user-1",
            chat_id="chat-1",
        )
        assert len(result.memories) == 1

    @pytest.mark.asyncio
    async def test_primary_search_handles_failure(self, mock_store):
        """Stage 1 should return empty list on failure."""
        mock_store.search = AsyncMock(side_effect=Exception("DB error"))

        pipeline = RetrievalPipeline(mock_store)
        context = RetrievalContext(
            user_id="user-1",
            query="test",
        )

        result = await pipeline.retrieve(context)

        assert result.memories == []

    @pytest.mark.asyncio
    async def test_finalize_deduplicates_memories(self, mock_store):
        """Stage 4 should deduplicate memories by ID."""
        # Return same memory twice with different similarities
        mock_store.search = AsyncMock(
            return_value=[
                SearchResult(
                    id="mem-1",
                    content="Test memory",
                    similarity=0.9,
                    metadata={},
                    source_type="memory",
                ),
                SearchResult(
                    id="mem-1",  # Duplicate ID
                    content="Test memory",
                    similarity=0.8,
                    metadata={},
                    source_type="memory",
                ),
            ]
        )

        pipeline = RetrievalPipeline(mock_store)
        context = RetrievalContext(
            user_id="user-1",
            query="test",
            max_memories=10,
        )

        result = await pipeline.retrieve(context)

        # Should only have one memory (deduplicated)
        assert len(result.memories) == 1
        assert result.memories[0].id == "mem-1"

    @pytest.mark.asyncio
    async def test_finalize_respects_max_memories(self, mock_store):
        """Stage 4 should limit results to max_memories."""
        mock_store.search = AsyncMock(
            return_value=[
                SearchResult(
                    id=f"mem-{i}",
                    content=f"Memory {i}",
                    similarity=0.9 - i * 0.1,
                    metadata={},
                    source_type="memory",
                )
                for i in range(10)
            ]
        )

        pipeline = RetrievalPipeline(mock_store)
        context = RetrievalContext(
            user_id="user-1",
            query="test",
            max_memories=3,
        )

        result = await pipeline.retrieve(context)

        assert len(result.memories) == 3


class TestPrivacyFilter:
    """Tests for privacy filtering in retrieval."""

    @pytest.fixture
    def pipeline(self, mock_store):
        return RetrievalPipeline(mock_store)

    def test_public_memory_passes_filter(self, pipeline):
        """PUBLIC memories should always pass."""
        result = pipeline._passes_privacy_filter(
            sensitivity=Sensitivity.PUBLIC,
            subject_person_ids=["person-1"],
            chat_type="group",
            querying_person_ids=set(),
        )
        assert result is True

    def test_none_sensitivity_passes_filter(self, pipeline):
        """None sensitivity should pass (treated as PUBLIC)."""
        result = pipeline._passes_privacy_filter(
            sensitivity=None,
            subject_person_ids=["person-1"],
            chat_type="group",
            querying_person_ids=set(),
        )
        assert result is True

    def test_personal_memory_passes_for_subject(self, pipeline):
        """PERSONAL memories should pass for the subject."""
        result = pipeline._passes_privacy_filter(
            sensitivity=Sensitivity.PERSONAL,
            subject_person_ids=["person-1"],
            chat_type="group",
            querying_person_ids={"person-1"},
        )
        assert result is True

    def test_personal_memory_fails_for_non_subject(self, pipeline):
        """PERSONAL memories should fail for non-subjects."""
        result = pipeline._passes_privacy_filter(
            sensitivity=Sensitivity.PERSONAL,
            subject_person_ids=["person-1"],
            chat_type="group",
            querying_person_ids={"person-2"},
        )
        assert result is False

    def test_sensitive_memory_passes_in_private_for_subject(self, pipeline):
        """SENSITIVE memories pass in private chat for subject."""
        result = pipeline._passes_privacy_filter(
            sensitivity=Sensitivity.SENSITIVE,
            subject_person_ids=["person-1"],
            chat_type="private",
            querying_person_ids={"person-1"},
        )
        assert result is True

    def test_sensitive_memory_fails_in_group(self, pipeline):
        """SENSITIVE memories fail in group chat even for subject."""
        result = pipeline._passes_privacy_filter(
            sensitivity=Sensitivity.SENSITIVE,
            subject_person_ids=["person-1"],
            chat_type="group",
            querying_person_ids={"person-1"},
        )
        assert result is False

    def test_sensitive_memory_fails_for_non_subject_in_private(self, pipeline):
        """SENSITIVE memories fail for non-subject even in private."""
        result = pipeline._passes_privacy_filter(
            sensitivity=Sensitivity.SENSITIVE,
            subject_person_ids=["person-1"],
            chat_type="private",
            querying_person_ids={"person-2"},
        )
        assert result is False


class TestStage1PrivacyFilter:
    """Tests for Stage 1 privacy filtering.

    Stage 1 returns the owner's own memories. The filter targets SENSITIVE
    memories about other people in group chats (health, medical, financial).
    Self-memories, PERSONAL notes, and PUBLIC facts pass through.
    """

    @pytest.fixture
    def pipeline(self, mock_store):
        return RetrievalPipeline(mock_store)

    @pytest.mark.asyncio
    async def test_stage1_sensitive_filtered_in_group(self, mock_store):
        """SENSITIVE memory about a non-participant excluded in group chat."""
        mock_store.search = AsyncMock(
            return_value=[
                SearchResult(
                    id="mem-sensitive",
                    content="Sarah is pregnant",
                    similarity=0.9,
                    metadata={
                        "sensitivity": "sensitive",
                        "subject_person_ids": ["person-sarah"],
                    },
                    source_type="memory",
                )
            ]
        )

        pipeline = RetrievalPipeline(mock_store)
        context = RetrievalContext(
            user_id="user-1",
            query="what do you know about people",
            chat_type="group",
            participant_person_ids={"bob": {"person-bob"}},
        )

        result = await pipeline.retrieve(context)
        assert len(result.memories) == 0

    @pytest.mark.asyncio
    async def test_stage1_sensitive_passes_for_participant_in_group(self, mock_store):
        """SENSITIVE memory passes in group when subject is a participant."""
        mock_store.search = AsyncMock(
            return_value=[
                SearchResult(
                    id="mem-sensitive",
                    content="Sarah is pregnant",
                    similarity=0.9,
                    metadata={
                        "sensitivity": "sensitive",
                        "subject_person_ids": ["person-sarah"],
                    },
                    source_type="memory",
                )
            ]
        )

        pipeline = RetrievalPipeline(mock_store)
        context = RetrievalContext(
            user_id="user-1",
            query="what do you know about Sarah",
            chat_type="group",
            participant_person_ids={"sarah": {"person-sarah"}},
        )

        result = await pipeline.retrieve(context)
        assert len(result.memories) == 1

    @pytest.mark.asyncio
    async def test_stage1_personal_passes_in_group(self, mock_store):
        """PERSONAL memory passes in group — owner's own notes are accessible."""
        mock_store.search = AsyncMock(
            return_value=[
                SearchResult(
                    id="mem-personal",
                    content="Looking for a new job",
                    similarity=0.85,
                    metadata={
                        "sensitivity": "personal",
                        "subject_person_ids": ["person-alice"],
                    },
                    source_type="memory",
                )
            ]
        )

        pipeline = RetrievalPipeline(mock_store)
        context = RetrievalContext(
            user_id="user-1",
            query="what do you know about Alice",
            chat_type="group",
            participant_person_ids={"bob": {"person-bob"}},
        )

        result = await pipeline.retrieve(context)
        assert len(result.memories) == 1

    @pytest.mark.asyncio
    async def test_stage1_public_passes_in_group(self, mock_store):
        """PUBLIC and None-sensitivity memories pass through in group chat."""
        mock_store.search = AsyncMock(
            return_value=[
                SearchResult(
                    id="mem-public",
                    content="Alice likes pizza",
                    similarity=0.9,
                    metadata={
                        "sensitivity": "public",
                        "subject_person_ids": ["person-alice"],
                    },
                    source_type="memory",
                ),
                SearchResult(
                    id="mem-none",
                    content="Bob works at Acme",
                    similarity=0.85,
                    metadata={
                        "sensitivity": None,
                        "subject_person_ids": ["person-bob"],
                    },
                    source_type="memory",
                ),
            ]
        )

        pipeline = RetrievalPipeline(mock_store)
        context = RetrievalContext(
            user_id="user-1",
            query="what do you know about people",
            chat_type="group",
            participant_person_ids={"alice": {"person-alice"}},
        )

        result = await pipeline.retrieve(context)
        assert len(result.memories) == 2

    @pytest.mark.asyncio
    async def test_stage1_sensitive_passes_in_private(self, mock_store):
        """SENSITIVE memory passes in private chat (filter only applies to groups)."""
        mock_store.search = AsyncMock(
            return_value=[
                SearchResult(
                    id="mem-sensitive",
                    content="Sarah is pregnant",
                    similarity=0.9,
                    metadata={
                        "sensitivity": "sensitive",
                        "subject_person_ids": ["person-sarah"],
                    },
                    source_type="memory",
                )
            ]
        )

        pipeline = RetrievalPipeline(mock_store)
        context = RetrievalContext(
            user_id="user-1",
            query="what do you know about Sarah",
            chat_type="private",
            participant_person_ids={"sarah": {"person-sarah"}},
        )

        result = await pipeline.retrieve(context)
        assert len(result.memories) == 1
        assert result.memories[0].id == "mem-sensitive"

    @pytest.mark.asyncio
    async def test_stage1_self_memory_passes_in_group(self, mock_store):
        """Self-memory (no subjects) always visible to owner in group chat."""
        mock_store.search = AsyncMock(
            return_value=[
                SearchResult(
                    id="mem-self",
                    content="I have anxiety",
                    similarity=0.9,
                    metadata={
                        "sensitivity": "sensitive",
                        "subject_person_ids": [],
                    },
                    source_type="memory",
                )
            ]
        )

        pipeline = RetrievalPipeline(mock_store)
        context = RetrievalContext(
            user_id="user-1",
            query="how am I doing",
            chat_type="group",
            participant_person_ids={"bob": {"person-bob"}},
        )

        result = await pipeline.retrieve(context)
        assert len(result.memories) == 1


class TestFinalizeRanking:
    """Tests for _finalize ranking behavior."""

    def test_dedup_keeps_highest_similarity(self, mock_store):
        """Dedup should keep the highest-similarity entry, not first-seen."""
        pipeline = RetrievalPipeline(mock_store)

        memories = [
            SearchResult(
                id="mem-1",
                content="Memory A",
                similarity=0.5,
                metadata={},
                source_type="memory",
            ),
            SearchResult(
                id="mem-1",  # Duplicate
                content="Memory A",
                similarity=0.9,
                metadata={},
                source_type="memory",
            ),
            SearchResult(
                id="mem-2",
                content="Memory B",
                similarity=0.7,
                metadata={},
                source_type="memory",
            ),
        ]

        result = pipeline._simple_finalize(memories, max_memories=10)

        assert len(result.memories) == 2
        # mem-1 should have the higher similarity (0.9)
        mem1 = next(m for m in result.memories if m.id == "mem-1")
        assert mem1.similarity == 0.9

    def test_results_sorted_by_similarity(self, mock_store):
        """Results should be sorted by similarity descending."""
        pipeline = RetrievalPipeline(mock_store)

        memories = [
            SearchResult(
                id="low",
                content="Low sim",
                similarity=0.3,
                metadata={},
                source_type="memory",
            ),
            SearchResult(
                id="high",
                content="High sim",
                similarity=0.95,
                metadata={},
                source_type="memory",
            ),
            SearchResult(
                id="mid",
                content="Mid sim",
                similarity=0.6,
                metadata={},
                source_type="memory",
            ),
        ]

        result = pipeline._simple_finalize(memories, max_memories=10)

        assert [m.id for m in result.memories] == ["high", "mid", "low"]


class TestRRFFusion:
    """Tests for RRF (Reciprocal Rank Fusion) in Stage 4."""

    def _make_result(self, id: str, similarity: float = 0.5) -> SearchResult:
        return SearchResult(
            id=id,
            content=f"Content of {id}",
            similarity=similarity,
            metadata={},
            source_type="memory",
        )

    def test_rrf_empty_stages(self, mock_store):
        """Empty stages should return empty results."""
        pipeline = RetrievalPipeline(mock_store)
        result = pipeline._rrf_finalize([[], [], []], max_memories=10)
        assert result.memories == []

    def test_rrf_single_stage_falls_back_to_simple(self, mock_store):
        """Single active stage should sort by similarity (no RRF)."""
        pipeline = RetrievalPipeline(mock_store)
        stage1 = [
            self._make_result("a", 0.3),
            self._make_result("b", 0.9),
            self._make_result("c", 0.6),
        ]
        result = pipeline._rrf_finalize([stage1, [], []], max_memories=10)
        assert [m.id for m in result.memories] == ["b", "c", "a"]

    def test_rrf_boosts_multi_stage_results(self, mock_store):
        """Results appearing in multiple stages should rank higher."""
        pipeline = RetrievalPipeline(mock_store)

        # mem-shared appears in both stages, mem-top only in stage1
        stage1 = [
            self._make_result("mem-top", 0.95),
            self._make_result("mem-shared", 0.8),
        ]
        stage2 = [
            self._make_result("mem-shared", 0.7),
            self._make_result("mem-only-s2", 0.5),
        ]
        result = pipeline._rrf_finalize([stage1, stage2], max_memories=10)

        ids = [m.id for m in result.memories]
        # mem-shared should be first due to RRF boost from appearing in both stages
        assert ids[0] == "mem-shared"

    def test_rrf_deduplicates(self, mock_store):
        """RRF should not produce duplicate entries."""
        pipeline = RetrievalPipeline(mock_store)

        stage1 = [self._make_result("a", 0.9)]
        stage2 = [self._make_result("a", 0.7)]
        result = pipeline._rrf_finalize([stage1, stage2], max_memories=10)

        assert len(result.memories) == 1
        assert result.memories[0].id == "a"
        # Should keep highest similarity version
        assert result.memories[0].similarity == 0.9

    def test_rrf_respects_max_memories(self, mock_store):
        """RRF should limit output to max_memories."""
        pipeline = RetrievalPipeline(mock_store)

        stage1 = [self._make_result(f"s1-{i}") for i in range(5)]
        stage2 = [self._make_result(f"s2-{i}") for i in range(5)]
        result = pipeline._rrf_finalize([stage1, stage2], max_memories=3)

        assert len(result.memories) == 3

    def test_rrf_scores_proportional_to_rank(self, mock_store):
        """Top-ranked items in a stage should have higher RRF contribution."""
        pipeline = RetrievalPipeline(mock_store)

        # Stage with 3 items: rank 0, 1, 2
        stage1 = [
            self._make_result("first", 0.9),
            self._make_result("second", 0.8),
            self._make_result("third", 0.7),
        ]
        # Stage2 only has 'third'
        stage2 = [self._make_result("third", 0.5)]

        result = pipeline._rrf_finalize([stage1, stage2], max_memories=10)

        # 'third' has RRF from both stages:
        #   stage1: 1/(60+3) = 1/63
        #   stage2: 1/(60+1) = 1/61
        #   total = 1/63 + 1/61
        #
        # 'first' has: 1/(60+1) = 1/61
        # 'second' has: 1/(60+2) = 1/62
        #
        # So 'third' (1/63 + 1/61) > 'first' (1/61) > 'second' (1/62)
        ids = [m.id for m in result.memories]
        assert ids[0] == "third"  # Boosted by appearing in both stages


class TestHybridSearch:
    """Tests for hybrid vector + person-graph search in store.search()."""

    @pytest.mark.asyncio
    async def test_search_by_person_name_returns_about_linked_memories(
        self, graph_store: Store, mock_index
    ):
        """Searching a person's name should return memories linked via ABOUT edges."""
        person = await graph_store.create_person(created_by="user-1", name="Alice")
        mem = await graph_store.add_memory(
            content="Alice's product launch is March 15",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        # Vector search returns nothing (low similarity)
        mock_index.search.return_value = []

        results = await graph_store.search("Alice", limit=5, owner_user_id="user-1")

        assert len(results) == 1
        assert results[0].id == mem.id
        assert results[0].similarity > 0

    @pytest.mark.asyncio
    async def test_search_follows_has_relationship_one_hop(
        self, graph_store: Store, mock_index
    ):
        """Search should follow HAS_RELATIONSHIP to find related people's memories."""
        alice = await graph_store.create_person(created_by="user-1", name="Alice")
        bob = await graph_store.create_person(created_by="user-1", name="Bob")
        await graph_store.add_relationship(
            alice.id, "business_partner", related_person_id=bob.id
        )
        mem = await graph_store.add_memory(
            content="Bob's keynote is on Friday",
            owner_user_id="user-1",
            subject_person_ids=[bob.id],
        )
        mock_index.search.return_value = []

        results = await graph_store.search("Alice", limit=5, owner_user_id="user-1")

        assert len(results) == 1
        assert results[0].id == mem.id
        # Related person memories get lower score
        assert results[0].similarity < 0.75

    @pytest.mark.asyncio
    async def test_vector_and_graph_results_merge_dedup(
        self, graph_store: Store, mock_index
    ):
        """Vector and graph results should merge, keeping the higher score."""
        person = await graph_store.create_person(created_by="user-1", name="Alice")
        mem = await graph_store.add_memory(
            content="Alice likes Italian food",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        # Vector search also finds this memory with high similarity
        mock_index.search.return_value = [(mem.id, 0.92)]

        results = await graph_store.search("Alice", limit=5, owner_user_id="user-1")

        # Should be deduplicated to one result
        assert len(results) == 1
        assert results[0].id == mem.id
        # Should keep the higher score (vector 0.92 > graph 0.75)
        assert results[0].similarity > 0.8

    @pytest.mark.asyncio
    async def test_no_person_match_falls_back_to_vector_only(
        self, graph_store: Store, mock_index
    ):
        """When query doesn't resolve to a person, only vector results are returned."""
        mem = await graph_store.add_memory(
            content="The sky is blue",
            owner_user_id="user-1",
        )
        mock_index.search.return_value = [(mem.id, 0.85)]

        results = await graph_store.search("sky color", limit=5, owner_user_id="user-1")

        assert len(results) == 1
        assert results[0].id == mem.id

    @pytest.mark.asyncio
    async def test_scope_filtering_applies_to_graph_results(
        self, graph_store: Store, mock_index
    ):
        """Graph results should respect scope filtering (owner_user_id)."""
        person = await graph_store.create_person(created_by="user-1", name="Alice")
        # Memory owned by user-2
        await graph_store.add_memory(
            content="Alice's secret project",
            owner_user_id="user-2",
            subject_person_ids=[person.id],
        )
        mock_index.search.return_value = []

        results = await graph_store.search("Alice", limit=5, owner_user_id="user-1")

        # user-1 should not see user-2's personal memory
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_archived_superseded_expired_excluded_from_graph(
        self, graph_store: Store, mock_index
    ):
        """Archived, superseded, and expired memories should be excluded from graph results."""
        from datetime import timedelta

        person = await graph_store.create_person(created_by="user-1", name="Alice")

        # Archived memory
        archived = await graph_store.add_memory(
            content="Alice old fact",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        graph_store.graph.memories[archived.id].archived_at = archived.created_at

        # Superseded memory
        superseded = await graph_store.add_memory(
            content="Alice outdated fact",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        graph_store.graph.memories[superseded.id].superseded_at = superseded.created_at

        # Expired memory
        from datetime import UTC, datetime

        await graph_store.add_memory(
            content="Alice temp fact",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
            expires_at=datetime.now(UTC) - timedelta(hours=1),
        )

        # Active memory
        active = await graph_store.add_memory(
            content="Alice current fact",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )

        mock_index.search.return_value = []

        results = await graph_store.search("Alice", limit=10, owner_user_id="user-1")

        assert len(results) == 1
        assert results[0].id == active.id

    @pytest.mark.asyncio
    async def test_search_by_alias(self, graph_store: Store, mock_index):
        """Searching by a person's alias should find their memories."""
        person = await graph_store.create_person(
            created_by="user-1", name="Alice", aliases=["Al"]
        )
        mem = await graph_store.add_memory(
            content="Alice enjoys hiking",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        mock_index.search.return_value = []

        results = await graph_store.search("Al", limit=5, owner_user_id="user-1")

        assert len(results) == 1
        assert results[0].id == mem.id
