"""Tests for RetrievalPipeline.

Tests the multi-stage memory retrieval logic.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.store.retrieval import (
    RetrievalContext,
    RetrievalPipeline,
)
from ash.store.types import SearchResult, Sensitivity


@pytest.fixture
def mock_store():
    """Create a mock Store for testing."""
    store = MagicMock()
    store.search = AsyncMock(return_value=[])
    store._resolve_subject_name = AsyncMock(return_value=None)
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

        result = pipeline._finalize(memories, max_memories=10)

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

        result = pipeline._finalize(memories, max_memories=10)

        assert [m.id for m in result.memories] == ["high", "mid", "low"]
