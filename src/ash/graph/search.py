"""Search and context retrieval mixin for GraphStore."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ash.memory.types import (
    MemoryEntry,
    RetrievedContext,
    SearchResult,
    Sensitivity,
    matches_scope,
)

if TYPE_CHECKING:
    from ash.graph.store import GraphStore

logger = logging.getLogger(__name__)

# Default similarity scores for non-vector results (used for ranking only)
_CROSS_CONTEXT_SIMILARITY = 0.7
_GRAPH_TRAVERSAL_SIMILARITY = 0.6


class SearchMixin:
    """Search, context retrieval, and privacy filtering."""

    async def search(
        self: GraphStore,
        query: str,
        limit: int = 5,
        subject_person_id: str | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[SearchResult]:
        vector_results = await self._index.search(query, limit=limit * 2)
        now = datetime.now(UTC)

        await self._ensure_graph_built()

        results: list[SearchResult] = []
        for vr in vector_results:
            memory = self._memory_by_id.get(vr.memory_id)
            if not memory:
                continue
            if memory.superseded_at:
                continue
            if memory.expires_at and memory.expires_at <= now:
                continue
            if not matches_scope(memory, owner_user_id, chat_id):
                continue
            if subject_person_id:
                if subject_person_id not in (memory.subject_person_ids or []):
                    continue

            subject_name = await self._resolve_subject_name(memory.subject_person_ids)
            metadata: dict[str, Any] = {
                "memory_type": memory.memory_type.value,
                "subject_person_ids": memory.subject_person_ids,
                "sensitivity": memory.sensitivity.value if memory.sensitivity else None,
                **(memory.metadata or {}),
            }
            if subject_name:
                metadata["subject_name"] = subject_name

            results.append(
                SearchResult(
                    id=memory.id,
                    content=memory.content,
                    similarity=vr.similarity,
                    metadata=metadata,
                    source_type="memory",
                )
            )
            if len(results) >= limit:
                break

        return results

    async def _find_memories_about_persons(
        self: GraphStore,
        person_ids: set[str],
        exclude_owner_user_id: str | None = None,
        limit: int = 20,
        portable_only: bool = True,
    ) -> list[MemoryEntry]:
        """Find memories about given persons using the GraphIndex.

        Uses O(1) graph lookups instead of scanning all memories.
        """
        graph = await self._ensure_graph_built()
        now = datetime.now(UTC)

        candidate_ids: set[str] = set()
        for pid in person_ids:
            candidate_ids |= graph.memories_about(pid)

        result: list[MemoryEntry] = []
        for mid in candidate_ids:
            memory = self._memory_by_id.get(mid)
            if not memory:
                continue
            if memory.archived_at is not None:
                continue
            if memory.expires_at and memory.expires_at <= now:
                continue
            if memory.superseded_at:
                continue
            if exclude_owner_user_id and memory.owner_user_id == exclude_owner_user_id:
                continue
            if portable_only and not memory.portable:
                continue
            result.append(memory)
            if len(result) >= limit:
                break

        return result

    def _passes_privacy_filter(
        self: GraphStore,
        sensitivity: Sensitivity | None,
        subject_person_ids: list[str],
        chat_type: str | None,
        querying_person_ids: set[str],
    ) -> bool:
        if sensitivity is None or sensitivity == Sensitivity.PUBLIC:
            return True
        is_subject = bool(set(subject_person_ids) & querying_person_ids)
        if sensitivity == Sensitivity.PERSONAL:
            return is_subject
        if sensitivity == Sensitivity.SENSITIVE:
            return chat_type == "private" and is_subject
        return False

    async def _make_cross_context_result(
        self: GraphStore,
        memory: MemoryEntry,
        similarity: float,
        *,
        graph_traversal: bool = False,
    ) -> SearchResult:
        """Convert a MemoryEntry to a SearchResult for cross-context retrieval."""
        subject_name = await self._resolve_subject_name(memory.subject_person_ids)
        meta: dict[str, Any] = {
            "memory_type": memory.memory_type.value,
            "subject_person_ids": memory.subject_person_ids,
            "cross_context": True,
        }
        if graph_traversal:
            meta["graph_traversal"] = True
        if subject_name:
            meta["subject_name"] = subject_name
        return SearchResult(
            id=memory.id,
            content=memory.content,
            similarity=similarity,
            metadata=meta,
            source_type="memory",
        )

    async def _collect_cross_context_memories(
        self: GraphStore,
        participant_person_ids: dict[str, set[str]],
        user_id: str,
        chat_type: str | None,
        max_memories: int,
    ) -> list[SearchResult]:
        """Retrieve memories from other contexts about chat participants."""
        results: list[SearchResult] = []
        for username, person_ids in participant_person_ids.items():
            if not person_ids:
                continue
            try:
                cross_memories = await self._find_memories_about_persons(
                    person_ids=person_ids,
                    exclude_owner_user_id=user_id,
                    limit=max_memories,
                )
                for memory in cross_memories:
                    if self._passes_privacy_filter(
                        sensitivity=memory.sensitivity,
                        subject_person_ids=memory.subject_person_ids,
                        chat_type=chat_type,
                        querying_person_ids=person_ids,
                    ):
                        results.append(
                            await self._make_cross_context_result(
                                memory, _CROSS_CONTEXT_SIMILARITY
                            )
                        )
            except Exception:
                logger.warning("Failed to get cross-context memories for %s", username)
        return results

    async def _collect_graph_traversal_memories(
        self: GraphStore,
        memories: list[SearchResult],
        seen_ids: set[str],
        participant_person_ids: dict[str, set[str]] | None,
        user_id: str,
        chat_type: str | None,
        max_memories: int,
    ) -> list[SearchResult]:
        """Find additional memories about people mentioned in existing results."""
        mentioned_person_ids: set[str] = set()
        for m in memories:
            spids = (m.metadata or {}).get("subject_person_ids") or []
            mentioned_person_ids.update(spids)

        # Exclude participants already handled by cross-context retrieval
        if participant_person_ids:
            for pids in participant_person_ids.values():
                mentioned_person_ids -= pids

        if not mentioned_person_ids:
            return []

        try:
            subject_cross = await self._find_memories_about_persons(
                person_ids=mentioned_person_ids,
                exclude_owner_user_id=user_id,
                limit=max_memories,
            )
        except Exception:
            logger.warning("Graph traversal cross-context failed", exc_info=True)
            return []

        all_participant_ids: set[str] = set()
        if participant_person_ids:
            for pids in participant_person_ids.values():
                all_participant_ids |= pids

        results: list[SearchResult] = []
        for memory in subject_cross:
            if memory.id in seen_ids:
                continue
            if self._passes_privacy_filter(
                sensitivity=memory.sensitivity,
                subject_person_ids=memory.subject_person_ids,
                chat_type=chat_type,
                querying_person_ids=all_participant_ids,
            ):
                results.append(
                    await self._make_cross_context_result(
                        memory,
                        _GRAPH_TRAVERSAL_SIMILARITY,
                        graph_traversal=True,
                    )
                )
                seen_ids.add(memory.id)
        return results

    async def get_context_for_message(
        self: GraphStore,
        user_id: str,
        user_message: str,
        chat_id: str | None = None,
        max_memories: int = 10,
        chat_type: str | None = None,
        participant_person_ids: dict[str, set[str]] | None = None,
    ) -> RetrievedContext:
        # Primary: vector search scoped to user/chat
        try:
            memories = await self.search(
                query=user_message,
                limit=max_memories,
                owner_user_id=user_id,
                chat_id=chat_id,
            )
        except Exception:
            logger.error(
                "Primary memory search failed, returning without context",
                exc_info=True,
            )
            memories = []

        # Cross-context: memories about participants from other chats
        if participant_person_ids:
            memories.extend(
                await self._collect_cross_context_memories(
                    participant_person_ids, user_id, chat_type, max_memories
                )
            )

        # Graph traversal: memories about people mentioned in results so far
        seen_ids: set[str] = {m.id for m in memories}
        if seen_ids:
            memories.extend(
                await self._collect_graph_traversal_memories(
                    memories,
                    seen_ids,
                    participant_person_ids,
                    user_id,
                    chat_type,
                    max_memories,
                )
            )

        # Deduplicate and limit
        unique: list[SearchResult] = []
        deduped: set[str] = set()
        for m in memories:
            if m.id not in deduped:
                deduped.add(m.id)
                unique.append(m)
                if len(unique) >= max_memories:
                    break

        return RetrievedContext(memories=unique)
