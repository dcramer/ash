"""Tests for NumpyVectorIndex: CRUD, search, persistence."""

from __future__ import annotations

import numpy as np
import pytest

from ash.graph.vectors import NumpyVectorIndex


def _vec(dim: int = 4, val: float = 1.0) -> list[float]:
    """Return a simple unit-direction vector."""
    v = [0.0] * dim
    v[0] = val
    return v


def _rand_vec(dim: int = 4, rng: np.random.Generator | None = None) -> list[float]:
    """Return a random non-zero vector."""
    if rng is None:
        rng = np.random.default_rng(42)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


# =============================================================================
# Basic Operations
# =============================================================================


class TestBasicOps:
    def test_add_and_count(self):
        idx = NumpyVectorIndex()
        idx.add("a", _vec())
        idx.add("b", _vec(val=2.0))
        assert idx.count == 2

    def test_has(self):
        idx = NumpyVectorIndex()
        idx.add("a", _vec())
        assert idx.has("a")
        assert not idx.has("b")

    def test_get_ids(self):
        idx = NumpyVectorIndex()
        idx.add("a", _vec())
        idx.add("b", _vec(val=-1.0))
        assert idx.get_ids() == {"a", "b"}

    def test_clear(self):
        idx = NumpyVectorIndex()
        idx.add("a", _vec())
        idx.add("b", _vec(val=-1.0))
        idx.clear()
        assert idx.count == 0
        assert not idx.has("a")
        assert idx.get_ids() == set()

    def test_zero_norm_rejected(self):
        idx = NumpyVectorIndex()
        idx.add("a", [0.0, 0.0, 0.0, 0.0])
        assert idx.count == 0
        assert not idx.has("a")


# =============================================================================
# Update
# =============================================================================


class TestUpdate:
    def test_update_in_materialized_matrix(self):
        """Updating a vector that's already in the materialized matrix works."""
        idx = NumpyVectorIndex()
        idx.add("a", [1.0, 0.0, 0.0, 0.0])
        # Force materialization
        idx._flush()

        # Update to point in opposite direction
        idx.add("a", [0.0, 1.0, 0.0, 0.0])
        assert idx.count == 1

        # Search should reflect the update
        results = idx.search([0.0, 1.0, 0.0, 0.0], limit=1)
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_update_in_pending_buffer(self):
        """Updating a vector still in the pending buffer works."""
        idx = NumpyVectorIndex()
        idx.add("a", [1.0, 0.0, 0.0, 0.0])
        # Don't flush — stays in pending
        idx.add("a", [0.0, 1.0, 0.0, 0.0])
        assert idx.count == 1

        results = idx.search([0.0, 1.0, 0.0, 0.0], limit=1)
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_search_reflects_update(self):
        idx = NumpyVectorIndex()
        idx.add("a", [1.0, 0.0, 0.0, 0.0])
        idx.add("b", [0.0, 1.0, 0.0, 0.0])
        idx._flush()

        # Update "a" to point same direction as "b"
        idx.add("a", [0.0, 1.0, 0.0, 0.0])
        results = idx.search([0.0, 1.0, 0.0, 0.0], limit=2)
        # Both should have similarity ~1.0
        for _, sim in results:
            assert sim == pytest.approx(1.0, abs=1e-5)


# =============================================================================
# Remove
# =============================================================================


class TestRemove:
    def test_swap_remove_correctness(self):
        """Swap-remove should maintain correct ID-to-index mapping."""
        idx = NumpyVectorIndex()
        idx.add("a", [1.0, 0.0, 0.0, 0.0])
        idx.add("b", [0.0, 1.0, 0.0, 0.0])
        idx.add("c", [0.0, 0.0, 1.0, 0.0])
        idx._flush()

        # Remove "a" (first) — "c" (last) should be swapped into index 0
        idx.remove("a")
        assert idx.count == 2
        assert not idx.has("a")
        assert idx.has("b")
        assert idx.has("c")

        # Search for "c" direction should find "c"
        results = idx.search([0.0, 0.0, 1.0, 0.0], limit=1)
        assert results[0][0] == "c"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_remove_only_element(self):
        idx = NumpyVectorIndex()
        idx.add("a", _vec())
        idx.remove("a")
        assert idx.count == 0
        assert idx.search(_vec()) == []

    def test_remove_nonexistent(self):
        """Removing a non-existent ID is a no-op."""
        idx = NumpyVectorIndex()
        idx.add("a", _vec())
        idx.remove("nonexistent")
        assert idx.count == 1

    def test_add_remove_add_cycle(self):
        idx = NumpyVectorIndex()
        idx.add("a", [1.0, 0.0, 0.0, 0.0])
        idx.remove("a")
        assert idx.count == 0

        idx.add("a", [0.0, 1.0, 0.0, 0.0])
        assert idx.count == 1
        results = idx.search([0.0, 1.0, 0.0, 0.0], limit=1)
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)


# =============================================================================
# Search
# =============================================================================


class TestSearch:
    def test_empty_index(self):
        idx = NumpyVectorIndex()
        assert idx.search(_vec()) == []

    def test_zero_query(self):
        idx = NumpyVectorIndex()
        idx.add("a", _vec())
        assert idx.search([0.0, 0.0, 0.0, 0.0]) == []

    def test_descending_similarity(self):
        idx = NumpyVectorIndex()
        idx.add("exact", [1.0, 0.0, 0.0, 0.0])
        idx.add("close", [0.9, 0.1, 0.0, 0.0])
        idx.add("far", [0.0, 1.0, 0.0, 0.0])

        results = idx.search([1.0, 0.0, 0.0, 0.0], limit=3)
        sims = [s for _, s in results]
        assert sims == sorted(sims, reverse=True)
        assert results[0][0] == "exact"

    def test_limit(self):
        idx = NumpyVectorIndex()
        rng = np.random.default_rng(123)
        for i in range(20):
            idx.add(f"v{i}", _rand_vec(rng=rng))

        results = idx.search(_rand_vec(rng=np.random.default_rng(0)), limit=5)
        assert len(results) == 5

    def test_partial_sort_path(self):
        """When limit < count, argpartition path is used."""
        idx = NumpyVectorIndex()
        rng = np.random.default_rng(42)
        for i in range(50):
            idx.add(f"v{i}", _rand_vec(rng=rng))

        results = idx.search(_rand_vec(rng=np.random.default_rng(0)), limit=5)
        assert len(results) == 5
        sims = [s for _, s in results]
        assert sims == sorted(sims, reverse=True)


# =============================================================================
# Persistence
# =============================================================================


class TestPersistence:
    async def test_save_load_roundtrip(self, tmp_path):
        idx = NumpyVectorIndex()
        idx.add("a", [1.0, 0.0, 0.0, 0.0])
        idx.add("b", [0.0, 1.0, 0.0, 0.0])

        path = tmp_path / "embeddings" / "test.npy"
        await idx.save(path)

        loaded = await NumpyVectorIndex.load(path)
        assert loaded.count == 2
        assert loaded.has("a")
        assert loaded.has("b")

        # Search should work on loaded index
        results = loaded.search([1.0, 0.0, 0.0, 0.0], limit=1)
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    async def test_both_files_created(self, tmp_path):
        idx = NumpyVectorIndex()
        idx.add("a", _vec())

        path = tmp_path / "test.npy"
        await idx.save(path)

        assert path.exists()
        assert path.with_suffix(".ids.json").exists()

    async def test_empty_save_removes_files(self, tmp_path):
        idx = NumpyVectorIndex()
        idx.add("a", _vec())

        path = tmp_path / "test.npy"
        await idx.save(path)
        assert path.exists()

        # Clear and save again — files should be removed
        idx.clear()
        await idx.save(path)
        assert not path.exists()
        assert not path.with_suffix(".ids.json").exists()

    async def test_size_mismatch_returns_empty(self, tmp_path, caplog):
        """If IDs and vectors have different lengths, return empty index."""
        import json
        import logging

        path = tmp_path / "test.npy"
        ids_path = path.with_suffix(".ids.json")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write mismatched data
        np.save(str(path), np.ones((3, 4), dtype=np.float32))
        with ids_path.open("w") as f:
            json.dump(["a", "b"], f)  # 2 ids but 3 vectors

        with caplog.at_level(logging.WARNING, logger="ash.graph.vectors"):
            loaded = await NumpyVectorIndex.load(path)

        assert loaded.count == 0
        assert "mismatch" in caplog.text.lower()

    async def test_overwrite_correctness(self, tmp_path):
        """Saving over an existing file produces correct results."""
        path = tmp_path / "test.npy"

        idx1 = NumpyVectorIndex()
        idx1.add("a", [1.0, 0.0, 0.0, 0.0])
        await idx1.save(path)

        idx2 = NumpyVectorIndex()
        idx2.add("x", [0.0, 1.0, 0.0, 0.0])
        idx2.add("y", [0.0, 0.0, 1.0, 0.0])
        await idx2.save(path)

        loaded = await NumpyVectorIndex.load(path)
        assert loaded.count == 2
        assert loaded.has("x")
        assert loaded.has("y")
        assert not loaded.has("a")
