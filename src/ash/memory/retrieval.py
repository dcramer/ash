"""Semantic search and retrieval using sqlite-vec."""

import json
import struct
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ash.memory.embeddings import EmbeddingGenerator


@dataclass
class SearchResult:
    """Search result with similarity score."""

    id: str
    content: str
    similarity: float
    metadata: dict[str, Any] | None = None
    source_type: str = "message"  # 'message' or 'knowledge'


class SemanticRetriever:
    """Semantic search over messages and knowledge using vector embeddings."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_generator: EmbeddingGenerator,
    ):
        """Initialize retriever.

        Args:
            session: Database session.
            embedding_generator: Embedding generator.
        """
        self._session = session
        self._embeddings = embedding_generator

    async def initialize_vector_tables(self) -> None:
        """Create sqlite-vec virtual tables if they don't exist.

        This should be called after database initialization.
        """
        dimensions = self._embeddings.dimensions

        # Create virtual tables for vector search
        await self._session.execute(
            text(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings USING vec0(
                    message_id TEXT PRIMARY KEY,
                    embedding FLOAT[{dimensions}]
                )
            """)
        )

        await self._session.execute(
            text(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_embeddings USING vec0(
                    knowledge_id TEXT PRIMARY KEY,
                    embedding FLOAT[{dimensions}]
                )
            """)
        )

        await self._session.commit()

    async def index_message(self, message_id: str, content: str) -> None:
        """Index a message for semantic search.

        Args:
            message_id: Message ID.
            content: Message content to embed.
        """
        embedding = await self._embeddings.embed(content)
        embedding_blob = self._serialize_embedding(embedding)

        # Delete existing embedding if any
        await self._session.execute(
            text("DELETE FROM message_embeddings WHERE message_id = :id"),
            {"id": message_id},
        )

        # Insert new embedding
        await self._session.execute(
            text(
                "INSERT INTO message_embeddings (message_id, embedding) VALUES (:id, :embedding)"
            ),
            {"id": message_id, "embedding": embedding_blob},
        )

    async def index_knowledge(self, knowledge_id: str, content: str) -> None:
        """Index a knowledge entry for semantic search.

        Args:
            knowledge_id: Knowledge ID.
            content: Knowledge content to embed.
        """
        embedding = await self._embeddings.embed(content)
        embedding_blob = self._serialize_embedding(embedding)

        # Delete existing embedding if any
        await self._session.execute(
            text("DELETE FROM knowledge_embeddings WHERE knowledge_id = :id"),
            {"id": knowledge_id},
        )

        # Insert new embedding
        await self._session.execute(
            text(
                "INSERT INTO knowledge_embeddings (knowledge_id, embedding) VALUES (:id, :embedding)"
            ),
            {"id": knowledge_id, "embedding": embedding_blob},
        )

    async def search_messages(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search messages by semantic similarity.

        Args:
            query: Search query.
            session_id: Optional session filter.
            limit: Maximum results.

        Returns:
            List of search results with similarity scores.
        """
        query_embedding = await self._embeddings.embed(query)
        embedding_blob = self._serialize_embedding(query_embedding)

        # Build query with optional session filter
        if session_id:
            sql = text("""
                SELECT
                    me.message_id,
                    m.content,
                    m.metadata,
                    vec_distance_cosine(me.embedding, :query_embedding) as distance
                FROM message_embeddings me
                JOIN messages m ON me.message_id = m.id
                WHERE m.session_id = :session_id
                ORDER BY distance ASC
                LIMIT :limit
            """)
            params = {
                "query_embedding": embedding_blob,
                "session_id": session_id,
                "limit": limit,
            }
        else:
            sql = text("""
                SELECT
                    me.message_id,
                    m.content,
                    m.metadata,
                    vec_distance_cosine(me.embedding, :query_embedding) as distance
                FROM message_embeddings me
                JOIN messages m ON me.message_id = m.id
                ORDER BY distance ASC
                LIMIT :limit
            """)
            params = {"query_embedding": embedding_blob, "limit": limit}

        result = await self._session.execute(sql, params)
        rows = result.fetchall()

        return [
            SearchResult(
                id=row[0],
                content=row[1],
                metadata=json.loads(row[2]) if row[2] else None,
                similarity=1.0 - row[3],  # Convert distance to similarity
                source_type="message",
            )
            for row in rows
        ]

    async def search_knowledge(
        self,
        query: str,
        limit: int = 10,
        include_expired: bool = False,
        subject_person_id: str | None = None,
    ) -> list[SearchResult]:
        """Search knowledge by semantic similarity.

        Args:
            query: Search query.
            limit: Maximum results.
            include_expired: Include expired entries.
            subject_person_id: Optional filter to knowledge about a specific person.

        Returns:
            List of search results with similarity scores.
        """
        query_embedding = await self._embeddings.embed(query)
        embedding_blob = self._serialize_embedding(query_embedding)

        # Build dynamic query with optional filters
        where_clauses = []
        params: dict[str, Any] = {
            "query_embedding": embedding_blob,
            "limit": limit,
        }

        if not include_expired:
            where_clauses.append(
                "(k.expires_at IS NULL OR k.expires_at > datetime('now'))"
            )

        if subject_person_id:
            where_clauses.append("k.subject_person_id = :subject_person_id")
            params["subject_person_id"] = subject_person_id

        where_clause = ""
        if where_clauses:
            where_clause = "WHERE " + " AND ".join(where_clauses)

        sql = text(f"""
            SELECT
                ke.knowledge_id,
                k.content,
                k.metadata,
                k.subject_person_id,
                p.name as subject_name,
                vec_distance_cosine(ke.embedding, :query_embedding) as distance
            FROM knowledge_embeddings ke
            JOIN knowledge k ON ke.knowledge_id = k.id
            LEFT JOIN people p ON k.subject_person_id = p.id
            {where_clause}
            ORDER BY distance ASC
            LIMIT :limit
        """)

        result = await self._session.execute(sql, params)
        rows = result.fetchall()

        return [
            SearchResult(
                id=row[0],
                content=row[1],
                metadata={
                    **(json.loads(row[2]) if row[2] else {}),
                    "subject_person_id": row[3],
                    "subject_name": row[4],
                },
                similarity=1.0 - row[5],  # Convert distance to similarity
                source_type="knowledge",
            )
            for row in rows
        ]

    async def search_all(
        self,
        query: str,
        limit: int = 10,
        subject_person_id: str | None = None,
    ) -> list[SearchResult]:
        """Search both messages and knowledge.

        Args:
            query: Search query.
            limit: Maximum results (combined).
            subject_person_id: Optional filter for knowledge about a specific person.

        Returns:
            List of search results sorted by similarity.
        """
        # Search both sources with limit
        messages = await self.search_messages(query, limit=limit)
        knowledge = await self.search_knowledge(
            query, limit=limit, subject_person_id=subject_person_id
        )

        # Combine and sort by similarity
        combined = messages + knowledge
        combined.sort(key=lambda x: x.similarity, reverse=True)

        return combined[:limit]

    async def delete_message_embedding(self, message_id: str) -> None:
        """Delete a message embedding.

        Args:
            message_id: Message ID.
        """
        await self._session.execute(
            text("DELETE FROM message_embeddings WHERE message_id = :id"),
            {"id": message_id},
        )

    async def delete_knowledge_embedding(self, knowledge_id: str) -> None:
        """Delete a knowledge embedding.

        Args:
            knowledge_id: Knowledge ID.
        """
        await self._session.execute(
            text("DELETE FROM knowledge_embeddings WHERE knowledge_id = :id"),
            {"id": knowledge_id},
        )

    def _serialize_embedding(self, embedding: list[float]) -> bytes:
        """Serialize embedding to bytes for sqlite-vec.

        Args:
            embedding: Embedding vector.

        Returns:
            Serialized bytes.
        """
        return struct.pack(f"{len(embedding)}f", *embedding)

    def _deserialize_embedding(self, data: bytes) -> list[float]:
        """Deserialize embedding from bytes.

        Args:
            data: Serialized bytes.

        Returns:
            Embedding vector.
        """
        count = len(data) // 4  # 4 bytes per float
        return list(struct.unpack(f"{count}f", data))
