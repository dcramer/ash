"""Vector index for semantic search.

Manages sqlite-vec virtual table for memory embeddings.
This is a rebuildable index - the source of truth is the JSONL file.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool

from ash.config.paths import get_database_path
from ash.memory.embeddings import EmbeddingGenerator

if TYPE_CHECKING:
    from ash.memory.types import MemoryEntry

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector search."""

    memory_id: str
    similarity: float


class VectorIndex:
    """Vector index for semantic memory search.

    Uses sqlite-vec for efficient similarity search. The index can be
    rebuilt from the JSONL source of truth if corrupted or missing.
    """

    def __init__(
        self,
        session: AsyncSession,
        embedding_generator: EmbeddingGenerator,
    ) -> None:
        """Initialize vector index.

        Args:
            session: Database session for sqlite-vec.
            embedding_generator: Generator for creating embeddings.
        """
        self._session = session
        self._embeddings = embedding_generator

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._embeddings.dimensions

    async def initialize(self) -> None:
        """Create sqlite-vec virtual table if it doesn't exist."""
        dimensions = self._embeddings.dimensions

        await self._session.execute(
            text(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
                    memory_id TEXT PRIMARY KEY,
                    embedding FLOAT[{dimensions}]
                )
            """)
        )
        await self._session.commit()

    async def add_embedding(
        self,
        memory_id: str,
        embedding: list[float] | bytes,
    ) -> None:
        """Add or update an embedding.

        Args:
            memory_id: Memory UUID.
            embedding: Embedding as float list or bytes.
        """
        if isinstance(embedding, list):
            embedding_blob = self._serialize_embedding(embedding)
        else:
            embedding_blob = embedding

        # Delete existing if any
        await self._session.execute(
            text("DELETE FROM memory_embeddings WHERE memory_id = :id"),
            {"id": memory_id},
        )

        # Insert new
        await self._session.execute(
            text(
                "INSERT INTO memory_embeddings (memory_id, embedding) VALUES (:id, :embedding)"
            ),
            {"id": memory_id, "embedding": embedding_blob},
        )

        await self._session.commit()

    async def delete_embedding(self, memory_id: str) -> None:
        """Delete an embedding.

        Args:
            memory_id: Memory UUID.
        """
        await self._session.execute(
            text("DELETE FROM memory_embeddings WHERE memory_id = :id"),
            {"id": memory_id},
        )
        await self._session.commit()

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[VectorSearchResult]:
        """Search by semantic similarity.

        Args:
            query: Text to search for.
            limit: Maximum results.

        Returns:
            List of results with similarity scores.
        """
        query_embedding = await self._embeddings.embed(query)
        return await self.search_by_embedding(query_embedding, limit)

    async def search_by_embedding(
        self,
        embedding: list[float],
        limit: int = 10,
    ) -> list[VectorSearchResult]:
        """Search by embedding vector.

        Args:
            embedding: Query embedding.
            limit: Maximum results.

        Returns:
            List of results with similarity scores.
        """
        embedding_blob = self._serialize_embedding(embedding)

        result = await self._session.execute(
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
                similarity=1.0 - row[1],  # Convert distance to similarity
            )
            for row in rows
        ]

    async def get_embedding_count(self) -> int:
        """Get number of indexed embeddings.

        Returns:
            Count of embeddings in index.
        """
        result = await self._session.execute(
            text("SELECT COUNT(*) FROM memory_embeddings")
        )
        return result.scalar() or 0

    async def clear(self) -> int:
        """Clear all embeddings from index.

        Returns:
            Number of embeddings deleted.
        """
        count = await self.get_embedding_count()
        await self._session.execute(text("DELETE FROM memory_embeddings"))
        await self._session.commit()
        return count

    async def rebuild_from_memories(
        self,
        memories: list[MemoryEntry],
    ) -> int:
        """Rebuild index from memory entries.

        Clears existing index and rebuilds from memory entries.
        Only indexes active (non-superseded) memories that have embeddings.

        Args:
            memories: Memory entries to index.

        Returns:
            Number of embeddings indexed.
        """
        # Clear existing
        await self.clear()

        # Index active memories with embeddings
        count = 0
        for memory in memories:
            if memory.superseded_at:
                continue  # Skip superseded

            embedding_bytes = memory.get_embedding_bytes()
            if not embedding_bytes:
                continue  # Skip without embedding

            await self._session.execute(
                text(
                    "INSERT INTO memory_embeddings (memory_id, embedding) VALUES (:id, :embedding)"
                ),
                {"id": memory.id, "embedding": embedding_bytes},
            )
            count += 1

        await self._session.commit()

        logger.info(
            "vector_index_rebuilt",
            extra={"indexed_count": count, "total_memories": len(memories)},
        )

        return count

    def _serialize_embedding(self, embedding: list[float]) -> bytes:
        """Serialize embedding to bytes for sqlite-vec."""
        return struct.pack(f"{len(embedding)}f", *embedding)


async def create_vector_index(
    db_path: Path | None = None,
    embedding_generator: EmbeddingGenerator | None = None,
) -> tuple[VectorIndex, AsyncSession]:
    """Create a vector index with its own database session.

    Args:
        db_path: Path to database (default: standard location).
        embedding_generator: Embedding generator (required if not rebuilding).

    Returns:
        Tuple of (VectorIndex, AsyncSession).
    """
    if db_path is None:
        db_path = get_database_path()

    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        poolclass=StaticPool,
    )

    from sqlalchemy.ext.asyncio import async_sessionmaker

    async_session_maker = async_sessionmaker(engine, expire_on_commit=False)
    session = async_session_maker()

    # Load sqlite-vec extension
    raw_conn = await session.connection()
    await raw_conn.run_sync(
        lambda conn: conn.connection.execute("SELECT 1")  # Ensure connection
    )

    if embedding_generator is None:
        # Create a minimal embedding generator for dimensions
        from ash.llm import LLMRegistry

        registry = LLMRegistry()
        embedding_generator = EmbeddingGenerator(registry)

    index = VectorIndex(session, embedding_generator)
    await index.initialize()

    return index, session


async def rebuild_vector_index_from_jsonl(
    memories_path: Path,
    db_path: Path | None = None,
) -> int:
    """Rebuild vector index from JSONL source of truth.

    This is a recovery operation - use when SQLite is missing or corrupted.

    Args:
        memories_path: Path to memories.jsonl.
        db_path: Path to database (default: standard location).

    Returns:
        Number of embeddings indexed.
    """
    from ash.memory.jsonl import MemoryJSONL

    if db_path is None:
        db_path = get_database_path()

    # Delete existing database if present
    if db_path.exists():
        db_path.unlink()
        logger.info("deleted_corrupted_db", extra={"path": str(db_path)})

    # Load memories from JSONL
    jsonl = MemoryJSONL(memories_path)
    memories = await jsonl.load_all()

    logger.info(
        "loaded_memories_for_rebuild",
        extra={"count": len(memories), "path": str(memories_path)},
    )

    # Create new index
    index, session = await create_vector_index(db_path)

    try:
        count = await index.rebuild_from_memories(memories)
        return count
    finally:
        await session.close()
