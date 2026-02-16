"""Numpy-based brute-force vector index.

Replaces sqlite-vec with simple numpy matmul for cosine similarity.
At Ash's scale (~thousands of memories, 1536-dim), this takes ~1-3ms.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class NumpyVectorIndex:
    """Brute-force cosine similarity using numpy.

    New vectors are buffered in a list and flushed into the main
    matrix lazily (before search, save, or remove) to avoid O(n)
    array copies on every add().
    """

    def __init__(self) -> None:
        self._vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._ids: list[str] = []
        self._id_to_index: dict[str, int] = {}
        self._pending: list[np.ndarray] = []

    def _flush(self) -> None:
        """Consolidate pending vectors into the main matrix."""
        if not self._pending:
            return
        new_block = np.stack(self._pending)
        if self._vectors.size == 0:
            self._vectors = new_block
        else:
            self._vectors = np.vstack([self._vectors, new_block])
        self._pending.clear()

    @property
    def count(self) -> int:
        return len(self._ids)

    def search(
        self, query_embedding: list[float], limit: int = 10
    ) -> list[tuple[str, float]]:
        """Return (id, similarity) pairs sorted by descending similarity."""
        if len(self._ids) == 0:
            return []

        self._flush()

        q = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        q /= norm

        scores = self._vectors @ q

        k = min(limit, len(self._ids))
        if k >= len(self._ids):
            # Sort all
            top_k = np.argsort(scores)[::-1][:k]
        else:
            # Partial sort for efficiency
            top_k = np.argpartition(scores, -k)[-k:]
            top_k = top_k[np.argsort(scores[top_k])[::-1]]

        return [(self._ids[i], float(scores[i])) for i in top_k]

    def add(self, node_id: str, embedding: list[float]) -> None:
        """Add or update a vector."""
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            logger.warning("Skipping zero-norm embedding for %s", node_id)
            return
        vec /= norm

        if node_id in self._id_to_index:
            idx = self._id_to_index[node_id]
            # If the vector is in the already-materialized matrix, update in place
            materialized_count = self._vectors.shape[0] if self._vectors.size > 0 else 0
            if idx < materialized_count:
                self._vectors[idx] = vec
            else:
                # Update in pending buffer
                self._pending[idx - materialized_count] = vec
            return

        idx = len(self._ids)
        self._ids.append(node_id)
        self._id_to_index[node_id] = idx
        self._pending.append(vec)

    def remove(self, node_id: str) -> None:
        """Remove a vector by ID."""
        idx = self._id_to_index.pop(node_id, None)
        if idx is None:
            return

        self._flush()

        last_idx = len(self._ids) - 1
        if idx != last_idx:
            # Swap with last element
            last_id = self._ids[last_idx]
            self._ids[idx] = last_id
            self._id_to_index[last_id] = idx
            self._vectors[idx] = self._vectors[last_idx]

        self._ids.pop()
        if len(self._ids) == 0:
            self._vectors = np.empty((0, 0), dtype=np.float32)
        else:
            self._vectors = self._vectors[: len(self._ids)]

    def clear(self) -> None:
        """Remove all vectors from the index."""
        self._ids.clear()
        self._id_to_index.clear()
        self._pending.clear()
        self._vectors = np.empty((0, 0), dtype=np.float32)

    def has(self, node_id: str) -> bool:
        """Check if a node has an embedding."""
        return node_id in self._id_to_index

    def get_ids(self) -> set[str]:
        """Get all indexed IDs."""
        return set(self._ids)

    async def save(self, path: Path) -> None:
        """Save to .npy + .ids.json"""
        import asyncio

        self._flush()
        await asyncio.to_thread(self._save_sync, path)

    def _save_sync(self, path: Path) -> None:
        """Synchronous save with atomic writes (runs in thread)."""
        import os
        import tempfile

        path.parent.mkdir(parents=True, exist_ok=True)
        ids_path = path.with_suffix(".ids.json")

        if len(self._ids) > 0:
            # Atomic write for .npy
            # suffix must be .npy so np.save doesn't append its own .npy extension
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npy")
            try:
                os.close(fd)
                np.save(tmp, self._vectors)
                # fsync the written file
                with Path(tmp).open("rb") as f:
                    os.fsync(f.fileno())
                Path(tmp).replace(path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

            # Atomic write for .ids.json
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(self._ids, f)
                    f.flush()
                    os.fsync(f.fileno())
                Path(tmp).replace(ids_path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        else:
            for p in [path, ids_path]:
                if p.exists():
                    p.unlink()

    @classmethod
    async def load(cls, path: Path) -> NumpyVectorIndex:
        """Load from .npy + .ids.json"""
        import asyncio

        return await asyncio.to_thread(cls._load_sync, path)

    @classmethod
    def _load_sync(cls, path: Path) -> NumpyVectorIndex:
        """Synchronous load (runs in thread)."""
        index = cls()
        ids_path = path.with_suffix(".ids.json")

        if path.exists() and ids_path.exists():
            index._vectors = np.load(str(path)).astype(np.float32)
            with ids_path.open() as f:
                index._ids = json.load(f)
            index._id_to_index = {id_: i for i, id_ in enumerate(index._ids)}

            if len(index._ids) != index._vectors.shape[0]:
                logger.warning(
                    "Vector index size mismatch: %d ids, %d vectors. Rebuilding.",
                    len(index._ids),
                    index._vectors.shape[0],
                )
                return cls()

        return index
