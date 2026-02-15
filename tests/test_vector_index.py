"""Integration tests for VectorIndex with real sqlite-vec.

These tests verify the actual sqlite-vec behavior rather than
testing through mocks.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.index import VectorIndex
from ash.memory.types import MemoryEntry, MemoryType

_SUPERSEDED_AT = datetime(2024, 1, 1, tzinfo=UTC)
_ARCHIVED_AT = datetime(2024, 1, 1, tzinfo=UTC)


@pytest.fixture
def mock_embedding_generator():
    """Embedding generator with fixed 4 dimensions for tests."""
    gen = MagicMock(spec=EmbeddingGenerator)
    gen.dimensions = 4
    gen.embed = AsyncMock()
    return gen


@pytest.fixture
async def db_session_with_vec(tmp_path):
    """Create a session with sqlite-vec loaded via Database class."""
    from ash.db.engine import Database

    db_path = tmp_path / "test_vec.db"
    db = Database(database_path=db_path)
    await db.connect()

    async with db.session() as session:
        yield session

    await db.disconnect()


@pytest.fixture
async def vector_index(
    db_session_with_vec: AsyncSession, mock_embedding_generator
) -> VectorIndex:
    """Create a VectorIndex with real sqlite-vec, 4 dimensions."""
    index = VectorIndex(db_session_with_vec, mock_embedding_generator)
    await index.initialize()
    return index


def _make_embedding(seed: float) -> list[float]:
    """Create a 4-dimensional embedding from a seed value."""
    return [seed, seed + 0.1, seed + 0.2, seed + 0.3]


class TestVectorIndexIntegration:
    """Integration tests for VectorIndex with real sqlite-vec."""

    async def test_add_and_search(self, vector_index: VectorIndex):
        """Adding embeddings and searching returns ranked results."""
        emb_a = _make_embedding(0.1)
        emb_b = _make_embedding(0.5)
        emb_c = _make_embedding(0.9)

        await vector_index.add_embedding("mem-a", emb_a)
        await vector_index.add_embedding("mem-b", emb_b)
        await vector_index.add_embedding("mem-c", emb_c)

        # Search with embedding close to emb_a
        results = await vector_index.search_by_embedding(emb_a, limit=3)

        assert len(results) == 3
        # Most similar should be mem-a itself
        assert results[0].memory_id == "mem-a"
        assert results[0].similarity > 0.99

    async def test_delete_removes_from_search(self, vector_index: VectorIndex):
        """Deleted embeddings no longer appear in search results."""
        emb_a = _make_embedding(0.1)
        emb_b = _make_embedding(0.5)

        await vector_index.add_embedding("mem-a", emb_a)
        await vector_index.add_embedding("mem-b", emb_b)

        await vector_index.delete_embedding("mem-a")

        results = await vector_index.search_by_embedding(emb_a, limit=5)
        result_ids = {r.memory_id for r in results}
        assert "mem-a" not in result_ids
        assert "mem-b" in result_ids

    async def test_delete_embeddings_batch(self, vector_index: VectorIndex):
        """Batch delete removes multiple embeddings."""
        await vector_index.add_embedding("mem-a", _make_embedding(0.1))
        await vector_index.add_embedding("mem-b", _make_embedding(0.3))
        await vector_index.add_embedding("mem-c", _make_embedding(0.5))

        await vector_index.delete_embeddings(["mem-a", "mem-b"])

        count = await vector_index.get_embedding_count()
        assert count == 1

    async def test_clear_removes_all(self, vector_index: VectorIndex):
        """Clear removes all embeddings."""
        await vector_index.add_embedding("mem-a", _make_embedding(0.1))
        await vector_index.add_embedding("mem-b", _make_embedding(0.5))

        count_before = await vector_index.get_embedding_count()
        assert count_before == 2

        deleted = await vector_index.clear()
        assert deleted == 2

        count_after = await vector_index.get_embedding_count()
        assert count_after == 0

    async def test_add_replaces_existing(self, vector_index: VectorIndex):
        """Adding with existing ID replaces the old embedding."""
        emb_v1 = _make_embedding(0.1)
        emb_v2 = _make_embedding(0.9)

        await vector_index.add_embedding("mem-a", emb_v1)
        await vector_index.add_embedding("mem-a", emb_v2)

        count = await vector_index.get_embedding_count()
        assert count == 1

        # Search should find the new embedding, not the old one
        results = await vector_index.search_by_embedding(emb_v2, limit=1)
        assert results[0].memory_id == "mem-a"
        assert results[0].similarity > 0.99

    async def test_rebuild_only_indexes_active_memories(
        self, vector_index: VectorIndex
    ):
        """rebuild_from_embeddings only indexes active (non-superseded, non-archived) memories."""

        active = MemoryEntry(
            id="mem-active",
            content="Active memory",
            memory_type=MemoryType.KNOWLEDGE,
        )
        superseded = MemoryEntry(
            id="mem-superseded",
            content="Superseded memory",
            memory_type=MemoryType.KNOWLEDGE,
            superseded_at=_SUPERSEDED_AT,
        )
        archived = MemoryEntry(
            id="mem-archived",
            content="Archived memory",
            memory_type=MemoryType.KNOWLEDGE,
            archived_at=_ARCHIVED_AT,
        )

        # Create base64-encoded embeddings
        emb = _make_embedding(0.5)
        emb_b64 = MemoryEntry.encode_embedding(emb)

        embeddings = {
            "mem-active": emb_b64,
            "mem-superseded": emb_b64,
            "mem-archived": emb_b64,
        }

        count = await vector_index.rebuild_from_embeddings(
            memories=[active, superseded, archived],
            embeddings=embeddings,
        )

        assert count == 1
        total = await vector_index.get_embedding_count()
        assert total == 1

    async def test_search_empty_index(self, vector_index: VectorIndex):
        """Searching an empty index returns empty results."""
        results = await vector_index.search_by_embedding(_make_embedding(0.5), limit=5)
        assert results == []

    async def test_similarity_ordering(self, vector_index: VectorIndex):
        """Results are ordered by similarity descending."""
        # Create embeddings at different distances from query
        query = [1.0, 0.0, 0.0, 0.0]
        close = [0.9, 0.1, 0.0, 0.0]
        medium = [0.5, 0.5, 0.0, 0.0]
        far = [0.0, 0.0, 1.0, 0.0]

        await vector_index.add_embedding("close", close)
        await vector_index.add_embedding("medium", medium)
        await vector_index.add_embedding("far", far)

        results = await vector_index.search_by_embedding(query, limit=3)

        assert len(results) == 3
        assert results[0].memory_id == "close"
        assert results[1].memory_id == "medium"
        assert results[2].memory_id == "far"
        # Similarities should be decreasing
        assert results[0].similarity > results[1].similarity > results[2].similarity
