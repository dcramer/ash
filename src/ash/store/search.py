"""Search and context retrieval mixin for Store (in-memory graph backed)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ash.store.retrieval import RetrievalContext, RetrievalPipeline
from ash.store.types import (
    RetrievedContext,
    SearchResult,
    matches_scope,
)

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)


class SearchMixin:
    """Search and context retrieval."""

    async def search(
        self: Store,
        query: str,
        limit: int = 5,
        subject_person_id: str | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[SearchResult]:
        try:
            query_embedding = await self._embeddings.embed(query)
        except Exception:
            logger.warning("Failed to generate query embedding", exc_info=True)
            return []

        vector_results = self._index.search(query_embedding, limit=limit * 2)
        now = datetime.now(UTC)

        results: list[SearchResult] = []
        for memory_id, similarity in vector_results:
            memory = self._graph.memories.get(memory_id)
            if not memory:
                continue
            if memory.superseded_at:
                continue
            if memory.archived_at is not None:
                continue
            if memory.expires_at and memory.expires_at <= now:
                continue
            if not matches_scope(memory, owner_user_id, chat_id):
                continue
            from ash.graph.edges import get_subject_person_ids
            from ash.store.trust import classify_trust, get_trust_weight

            mem_subjects = get_subject_person_ids(self._graph, memory.id)

            if subject_person_id:
                if subject_person_id not in mem_subjects:
                    continue

            # Apply trust weighting to similarity score
            trust_level = classify_trust(self._graph, memory.id)
            weighted_similarity = similarity * get_trust_weight(trust_level)

            subject_name = await self._resolve_subject_name(mem_subjects)
            metadata: dict[str, Any] = {
                "memory_type": memory.memory_type.value,
                "subject_person_ids": mem_subjects,
                "source_username": memory.source_username,
                "sensitivity": memory.sensitivity.value if memory.sensitivity else None,
                "trust": trust_level,
                **(memory.metadata or {}),
            }
            if subject_name:
                metadata["subject_name"] = subject_name

            results.append(
                SearchResult(
                    id=memory.id,
                    content=memory.content,
                    similarity=weighted_similarity,
                    metadata=metadata,
                    source_type="memory",
                )
            )
            if len(results) >= limit:
                break

        return results

    async def get_context_for_message(
        self: Store,
        user_id: str,
        user_message: str,
        chat_id: str | None = None,
        max_memories: int = 10,
        chat_type: str | None = None,
        participant_person_ids: dict[str, set[str]] | None = None,
    ) -> RetrievedContext:
        """Get context for a message using the retrieval pipeline."""
        pipeline = RetrievalPipeline(self)
        context = RetrievalContext(
            user_id=user_id,
            query=user_message,
            chat_id=chat_id,
            max_memories=max_memories,
            chat_type=chat_type,
            participant_person_ids=participant_person_ids or {},
        )
        return await pipeline.retrieve(context)
