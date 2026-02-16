"""Tests for NumpyVectorIndex.

These tests verify the numpy-based in-memory vector index used for
cosine similarity search.
"""

from pathlib import Path

import pytest

from ash.graph.vectors import NumpyVectorIndex


@pytest.fixture
def index() -> NumpyVectorIndex:
    return NumpyVectorIndex()


def _make_embedding(seed: float) -> list[float]:
    """Create a 4-dimensional embedding from a seed value."""
    return [seed, seed + 0.1, seed + 0.2, seed + 0.3]


class TestNumpyVectorIndex:
    async def test_add_and_search(self, index: NumpyVectorIndex):
        """Adding embeddings and searching returns ranked results."""
        emb_a = _make_embedding(0.1)
        emb_b = _make_embedding(0.5)
        emb_c = _make_embedding(0.9)

        index.add("mem-a", emb_a)
        index.add("mem-b", emb_b)
        index.add("mem-c", emb_c)

        # Search with embedding close to emb_a
        results = index.search(emb_a, limit=3)

        assert len(results) == 3
        # Most similar should be mem-a itself
        assert results[0][0] == "mem-a"
        assert results[0][1] > 0.99

    async def test_delete_removes_from_search(self, index: NumpyVectorIndex):
        """Deleted embeddings no longer appear in search results."""
        emb_a = _make_embedding(0.1)
        emb_b = _make_embedding(0.5)

        index.add("mem-a", emb_a)
        index.add("mem-b", emb_b)

        index.remove("mem-a")

        results = index.search(emb_a, limit=5)
        result_ids = {r[0] for r in results}
        assert "mem-a" not in result_ids
        assert "mem-b" in result_ids

    async def test_add_replaces_existing(self, index: NumpyVectorIndex):
        """Adding with existing ID replaces the old embedding."""
        emb_v1 = _make_embedding(0.1)
        emb_v2 = _make_embedding(0.9)

        index.add("mem-a", emb_v1)
        index.add("mem-a", emb_v2)

        assert index.count == 1

        # Search should find the new embedding, not the old one
        results = index.search(emb_v2, limit=1)
        assert results[0][0] == "mem-a"
        assert results[0][1] > 0.99

    async def test_search_empty_index(self, index: NumpyVectorIndex):
        """Searching an empty index returns empty results."""
        results = index.search(_make_embedding(0.5), limit=5)
        assert results == []

    async def test_similarity_ordering(self, index: NumpyVectorIndex):
        """Results are ordered by similarity descending."""
        query = [1.0, 0.0, 0.0, 0.0]
        close = [0.9, 0.1, 0.0, 0.0]
        medium = [0.5, 0.5, 0.0, 0.0]
        far = [0.0, 0.0, 1.0, 0.0]

        index.add("close", close)
        index.add("medium", medium)
        index.add("far", far)

        results = index.search(query, limit=3)

        assert len(results) == 3
        assert results[0][0] == "close"
        assert results[1][0] == "medium"
        assert results[2][0] == "far"
        # Similarities should be decreasing
        assert results[0][1] > results[1][1] > results[2][1]

    async def test_save_and_load_round_trip(
        self, index: NumpyVectorIndex, tmp_path: Path
    ):
        """Saving and loading preserves the index contents."""
        index.add("mem-a", _make_embedding(0.1))
        index.add("mem-b", _make_embedding(0.5))
        index.add("mem-c", _make_embedding(0.9))

        save_path = tmp_path / "vectors.npy"
        await index.save(save_path)

        loaded = await NumpyVectorIndex.load(save_path)

        assert loaded.count == 3

        # Verify search still works and returns same results
        query = _make_embedding(0.1)
        original_results = index.search(query, limit=3)
        loaded_results = loaded.search(query, limit=3)

        assert [r[0] for r in original_results] == [r[0] for r in loaded_results]
        for orig, load in zip(original_results, loaded_results, strict=False):
            assert abs(orig[1] - load[1]) < 1e-6

    async def test_count_after_add_and_remove(self, index: NumpyVectorIndex):
        """Count reflects current number of stored embeddings."""
        assert index.count == 0

        index.add("mem-a", _make_embedding(0.1))
        assert index.count == 1

        index.add("mem-b", _make_embedding(0.5))
        assert index.count == 2

        index.add("mem-c", _make_embedding(0.9))
        assert index.count == 3

        index.remove("mem-b")
        assert index.count == 2

        index.remove("mem-a")
        assert index.count == 1

        index.remove("mem-c")
        assert index.count == 0
