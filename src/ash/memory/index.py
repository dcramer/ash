"""Vector index for semantic search.

Manages sqlite-vec virtual table for memory embeddings.
This is a rebuildable index - the source of truth is the SQLite tables.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

from sqlalchemy import text

from ash.db.engine import Database
from ash.memory.embeddings import EmbeddingGenerator
from ash.store.types import MemoryEntry

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector search."""

    memory_id: str
    similarity: float


class VectorIndex:
    """Vector index for semantic memory search.

    Uses sqlite-vec for efficient similarity search.
    """

    def __init__(
        self,
        db: Database,
        embedding_generator: EmbeddingGenerator,
    ) -> None:
        self._db = db
        self._embeddings = embedding_generator

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._embeddings.dimensions

    async def initialize(self) -> None:
        """Create sqlite-vec virtual table if it doesn't exist."""
        dimensions = self._embeddings.dimensions

        async with self._db.session() as session:
            await session.execute(
                text(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
                        memory_id TEXT PRIMARY KEY,
                        embedding FLOAT[{dimensions}]
                    )
                """)
            )

    async def add_embedding(
        self,
        memory_id: str,
        embedding: list[float] | bytes,
    ) -> None:
        """Add or update an embedding."""
        if isinstance(embedding, list):
            embedding_blob = MemoryEntry.serialize_embedding_bytes(embedding)
        else:
            embedding_blob = embedding

        async with self._db.session() as session:
            await session.execute(
                text("DELETE FROM memory_embeddings WHERE memory_id = :id"),
                {"id": memory_id},
            )
            await session.execute(
                text(
                    "INSERT INTO memory_embeddings (memory_id, embedding) VALUES (:id, :embedding)"
                ),
                {"id": memory_id, "embedding": embedding_blob},
            )

    async def delete_embedding(self, memory_id: str) -> None:
        """Delete an embedding."""
        async with self._db.session() as session:
            await session.execute(
                text("DELETE FROM memory_embeddings WHERE memory_id = :id"),
                {"id": memory_id},
            )

    async def delete_embeddings(self, memory_ids: list[str]) -> None:
        """Delete embeddings for multiple memories."""
        if not memory_ids:
            return
        async with self._db.session() as session:
            for memory_id in memory_ids:
                await session.execute(
                    text("DELETE FROM memory_embeddings WHERE memory_id = :id"),
                    {"id": memory_id},
                )

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[VectorSearchResult]:
        """Search by semantic similarity."""
        query_embedding = await self._embeddings.embed(query)
        return await self.search_by_embedding(query_embedding, limit)

    async def search_by_embedding(
        self,
        embedding: list[float],
        limit: int = 10,
    ) -> list[VectorSearchResult]:
        """Search by embedding vector."""
        embedding_blob = MemoryEntry.serialize_embedding_bytes(embedding)

        async with self._db.session() as session:
            result = await session.execute(
                text("""
                    SELECT
                        memory_id,
                        vec_distance_cosine(embedding, :query_embedding) as distance
                    FROM memory_embeddings
                    ORDER BY distance ASC
                    LIMIT :limit
                """),
                {"query_embedding": embedding_blob, "limit": limit},
            )

            rows = result.fetchall()
            return [
                VectorSearchResult(
                    memory_id=row[0],
                    similarity=1.0 - row[1],
                )
                for row in rows
            ]

    async def get_embedding_count(self) -> int:
        """Get number of indexed embeddings."""
        async with self._db.session() as session:
            result = await session.execute(
                text("SELECT COUNT(*) FROM memory_embeddings")
            )
            return result.scalar() or 0

    async def clear(self) -> int:
        """Clear all embeddings from index."""
        count = await self.get_embedding_count()
        async with self._db.session() as session:
            await session.execute(text("DELETE FROM memory_embeddings"))
        return count

    async def rebuild_from_embeddings(
        self,
        memories: list[MemoryEntry],
        embeddings: dict[str, str],
    ) -> int:
        """Rebuild index from separate embeddings mapping.

        Clears existing index and rebuilds. Only indexes active
        (non-superseded, non-archived) memories that have embeddings.
        """
        await self.clear()

        active_ids = {
            m.id for m in memories if not m.superseded_at and m.archived_at is None
        }

        count = 0
        async with self._db.session() as session:
            for memory_id, embedding_b64 in embeddings.items():
                if memory_id not in active_ids:
                    continue
                if not embedding_b64:
                    continue

                embedding_bytes = base64.b64decode(embedding_b64)
                await session.execute(
                    text(
                        "INSERT INTO memory_embeddings (memory_id, embedding) VALUES (:id, :embedding)"
                    ),
                    {"id": memory_id, "embedding": embedding_bytes},
                )
                count += 1

        logger.info(
            "vector_index_rebuilt",
            extra={"indexed_count": count, "total_memories": len(memories)},
        )

        return count
