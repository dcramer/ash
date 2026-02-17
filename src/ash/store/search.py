"""Search and context retrieval mixin for Store (in-memory graph backed)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ash.graph.edges import (
    get_memories_about_person,
    get_related_people,
    get_subject_person_ids,
)
from ash.store.retrieval import RetrievalContext, RetrievalPipeline
from ash.store.trust import classify_trust, get_trust_weight
from ash.store.types import (
    RetrievedContext,
    SearchResult,
    matches_scope,
)

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)

# Default similarity scores for person-graph results
PERSON_GRAPH_DIRECT_SIMILARITY = 0.75
PERSON_GRAPH_RELATED_SIMILARITY = 0.55


class SearchMixin:
    """Search and context retrieval."""

    async def _resolve_query_to_person_ids(self: Store, query: str) -> set[str]:
        """Resolve a search query to person IDs via username, name, or alias."""
        # Try username-based lookup first (covers usernames, person names, aliases)
        person_ids = await self.find_person_ids_for_username(query)
        if person_ids:
            return person_ids

        # Fall back to find_person which also checks relationship terms
        person = await self.find_person(query)
        if person:
            return {person.id}

        return set()

    async def _graph_memories_for_persons(
        self: Store,
        person_ids: set[str],
        limit: int,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[SearchResult]:
        """Collect memories about persons via ABOUT edges and 1-hop relationships."""
        now = datetime.now(UTC)
        results_by_id: dict[str, SearchResult] = {}

        # Direct ABOUT memories for each person
        for pid in person_ids:
            for memory_id in get_memories_about_person(self._graph, pid):
                if memory_id in results_by_id:
                    continue
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

                trust_level = classify_trust(self._graph, memory.id)
                weighted = PERSON_GRAPH_DIRECT_SIMILARITY * get_trust_weight(
                    trust_level
                )
                mem_subjects = get_subject_person_ids(self._graph, memory.id)
                subject_name = await self._resolve_subject_name(mem_subjects)

                metadata: dict[str, Any] = {
                    "memory_type": memory.memory_type.value,
                    "subject_person_ids": mem_subjects,
                    "source_username": memory.source_username,
                    "sensitivity": (
                        memory.sensitivity.value if memory.sensitivity else None
                    ),
                    "trust": trust_level,
                    "discovery_stage": "person_graph",
                    **(memory.metadata or {}),
                }
                if subject_name:
                    metadata["subject_name"] = subject_name

                results_by_id[memory.id] = SearchResult(
                    id=memory.id,
                    content=memory.content,
                    similarity=weighted,
                    metadata=metadata,
                    source_type="memory",
                )

        # 1-hop: memories about related people via HAS_RELATIONSHIP
        related_pids: set[str] = set()
        for pid in person_ids:
            for related_pid in get_related_people(self._graph, pid):
                if related_pid not in person_ids:
                    related_pids.add(related_pid)

        for pid in related_pids:
            for memory_id in get_memories_about_person(self._graph, pid):
                if memory_id in results_by_id:
                    continue
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

                trust_level = classify_trust(self._graph, memory.id)
                weighted = PERSON_GRAPH_RELATED_SIMILARITY * get_trust_weight(
                    trust_level
                )
                mem_subjects = get_subject_person_ids(self._graph, memory.id)
                subject_name = await self._resolve_subject_name(mem_subjects)

                metadata = {
                    "memory_type": memory.memory_type.value,
                    "subject_person_ids": mem_subjects,
                    "source_username": memory.source_username,
                    "sensitivity": (
                        memory.sensitivity.value if memory.sensitivity else None
                    ),
                    "trust": trust_level,
                    "discovery_stage": "person_graph_related",
                    **(memory.metadata or {}),
                }
                if subject_name:
                    metadata["subject_name"] = subject_name

                results_by_id[memory.id] = SearchResult(
                    id=memory.id,
                    content=memory.content,
                    similarity=weighted,
                    metadata=metadata,
                    source_type="memory",
                )

        # Sort by similarity descending and limit
        sorted_results = sorted(
            results_by_id.values(), key=lambda r: r.similarity, reverse=True
        )
        return sorted_results[:limit]

    async def search(
        self: Store,
        query: str,
        limit: int = 5,
        subject_person_id: str | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[SearchResult]:
        # Vector search
        try:
            query_embedding = await self._embeddings.embed(query)
        except Exception:
            logger.warning("query_embedding_failed", exc_info=True)
            return []

        vector_results = self._index.search(query_embedding, limit=limit * 2)
        now = datetime.now(UTC)

        results_by_id: dict[str, SearchResult] = {}
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

            results_by_id[memory.id] = SearchResult(
                id=memory.id,
                content=memory.content,
                similarity=weighted_similarity,
                metadata=metadata,
                source_type="memory",
            )
            if len(results_by_id) >= limit * 2:
                break

        # Person-graph search: resolve query to person IDs and fetch graph memories
        try:
            person_ids = await self._resolve_query_to_person_ids(query)
            if person_ids:
                graph_results = await self._graph_memories_for_persons(
                    person_ids=person_ids,
                    limit=limit,
                    owner_user_id=owner_user_id,
                    chat_id=chat_id,
                )
                for result in graph_results:
                    existing = results_by_id.get(result.id)
                    if existing is None or result.similarity > existing.similarity:
                        results_by_id[result.id] = result
        except Exception:
            logger.warning("person_graph_search_failed", exc_info=True)

        # Sort by similarity and limit
        sorted_results = sorted(
            results_by_id.values(), key=lambda r: r.similarity, reverse=True
        )
        return sorted_results[:limit]

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
