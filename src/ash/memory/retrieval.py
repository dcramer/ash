"""Semantic search and retrieval using sqlite-vec.

Note: Message search has been removed since sessions/messages are now stored
in JSONL files. This module now only handles semantic search over memories.
"""

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
    source_type: str = "memory"


class SemanticRetriever:
    """Semantic search over memories using vector embeddings."""

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

        # Create virtual table for memory embeddings
        await self._session.execute(
            text(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
                    memory_id TEXT PRIMARY KEY,
                    embedding FLOAT[{dimensions}]
                )
            """)
        )

        await self._session.commit()

    async def index_memory(self, memory_id: str, content: str) -> None:
        """Index a memory entry for semantic search.

        Args:
            memory_id: Memory ID.
            content: Memory content to embed.
        """
        embedding = await self._embeddings.embed(content)
        embedding_blob = self._serialize_embedding(embedding)

        # Delete existing embedding if any
        await self._session.execute(
            text("DELETE FROM memory_embeddings WHERE memory_id = :id"),
            {"id": memory_id},
        )

        # Insert new embedding
        await self._session.execute(
            text(
                "INSERT INTO memory_embeddings (memory_id, embedding) VALUES (:id, :embedding)"
            ),
            {"id": memory_id, "embedding": embedding_blob},
        )

        # Commit to persist the embedding
        await self._session.commit()

    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        include_expired: bool = False,
        include_superseded: bool = False,
        subject_person_id: str | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[SearchResult]:
        """Search memories by semantic similarity.

        Memory scoping:
        - Personal: owner_user_id set - only visible to that user
        - Group: owner_user_id NULL, chat_id set - visible to everyone in that chat

        Args:
            query: Search query.
            limit: Maximum results.
            include_expired: Include expired entries.
            include_superseded: Include superseded entries.
            subject_person_id: Optional filter to memories about a specific person.
            owner_user_id: Filter to user's personal memories.
            chat_id: Filter to include group memories for this chat.

        Returns:
            List of search results with similarity scores and resolved subject_name.
        """
        query_embedding = await self._embeddings.embed(query)
        embedding_blob = self._serialize_embedding(query_embedding)

        # Build dynamic query with optional filters
        where_clauses: list[str] = []
        params: dict[str, Any] = {
            "query_embedding": embedding_blob,
            "limit": limit,
        }

        if not include_expired:
            where_clauses.append(
                "(m.expires_at IS NULL OR m.expires_at > datetime('now'))"
            )

        if not include_superseded:
            where_clauses.append("m.superseded_at IS NULL")

        if subject_person_id:
            # Use JSON function to check if person_id is in the array
            where_clauses.append(
                "EXISTS (SELECT 1 FROM json_each(m.subject_person_ids) "
                "WHERE json_each.value = :subject_person_id)"
            )
            params["subject_person_id"] = subject_person_id

        # Memory visibility scoping
        if owner_user_id or chat_id:
            visibility_conditions: list[str] = []

            if owner_user_id:
                # User's personal memories
                visibility_conditions.append("m.owner_user_id = :owner_user_id")
                params["owner_user_id"] = owner_user_id

            if chat_id:
                # Group memories for this chat (owner_user_id is NULL, chat_id matches)
                visibility_conditions.append(
                    "(m.owner_user_id IS NULL AND m.chat_id = :chat_id)"
                )
                params["chat_id"] = chat_id

            where_clauses.append(f"({' OR '.join(visibility_conditions)})")

        where_clause = ""
        if where_clauses:
            where_clause = "WHERE " + " AND ".join(where_clauses)

        sql = text(f"""
            SELECT
                me.memory_id,
                m.content,
                m.metadata,
                m.subject_person_ids,
                vec_distance_cosine(me.embedding, :query_embedding) as distance
            FROM memory_embeddings me
            JOIN memories m ON me.memory_id = m.id
            {where_clause}
            ORDER BY distance ASC
            LIMIT :limit
        """)  # noqa: S608 - where_clause is built from hardcoded conditions

        result = await self._session.execute(sql, params)
        rows = result.fetchall()

        # Collect all unique person IDs for name resolution
        all_person_ids: set[str] = set()
        for row in rows:
            subject_ids = json.loads(row[3]) if row[3] else None
            if subject_ids:
                all_person_ids.update(subject_ids)

        # Resolve person IDs to names
        person_names: dict[str, str] = {}
        if all_person_ids:
            person_names = await self._resolve_person_names(list(all_person_ids))

        # Build results with resolved subject names
        results: list[SearchResult] = []
        for row in rows:
            subject_ids = json.loads(row[3]) if row[3] else None
            base_metadata = json.loads(row[2]) if row[2] else {}

            # Build subject_name from resolved person IDs
            subject_name = None
            if subject_ids:
                names = [
                    person_names[pid] for pid in subject_ids if pid in person_names
                ]
                if names:
                    subject_name = ", ".join(names)

            results.append(
                SearchResult(
                    id=row[0],
                    content=row[1],
                    metadata={
                        **base_metadata,
                        "subject_person_ids": subject_ids,
                        "subject_name": subject_name,
                    },
                    similarity=1.0 - row[4],  # Convert distance to similarity
                    source_type="memory",
                )
            )

        return results

    async def _resolve_person_names(self, person_ids: list[str]) -> dict[str, str]:
        """Resolve person IDs to names.

        Args:
            person_ids: List of person UUIDs to resolve.

        Returns:
            Dict mapping person_id to name.
        """
        if not person_ids:
            return {}

        # Build parameterized query for batch lookup
        placeholders = ", ".join(f":id{i}" for i in range(len(person_ids)))
        params = {f"id{i}": pid for i, pid in enumerate(person_ids)}

        sql = text(f"""
            SELECT id, name FROM people WHERE id IN ({placeholders})
        """)  # noqa: S608 - placeholders built from indices

        result = await self._session.execute(sql, params)
        rows = result.fetchall()

        return {row[0]: row[1] for row in rows}

    # Alias for backward compatibility
    search = search_memories

    async def delete_memory_embedding(self, memory_id: str) -> None:
        """Delete a memory embedding.

        Args:
            memory_id: Memory ID.
        """
        await self._session.execute(
            text("DELETE FROM memory_embeddings WHERE memory_id = :id"),
            {"id": memory_id},
        )
        await self._session.commit()

    def _serialize_embedding(self, embedding: list[float]) -> bytes:
        """Serialize embedding to bytes for sqlite-vec."""
        return struct.pack(f"{len(embedding)}f", *embedding)
