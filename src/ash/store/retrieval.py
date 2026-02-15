"""Memory retrieval pipeline with clear stages.

Extracts the multi-stage retrieval logic from SearchMixin into a dedicated class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from ash.store.mappers import row_to_memory as _row_to_memory
from ash.store.memories.helpers import load_subjects as _load_subjects
from ash.store.types import (
    MemoryEntry,
    RetrievedContext,
    SearchResult,
    Sensitivity,
)

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)

# Default similarity scores for non-vector results (used for ranking only)
CROSS_CONTEXT_SIMILARITY = 0.7
GRAPH_TRAVERSAL_SIMILARITY = 0.6


@dataclass
class RetrievalContext:
    """Context for a retrieval operation."""

    user_id: str
    query: str
    chat_id: str | None = None
    max_memories: int = 10
    chat_type: str | None = None
    participant_person_ids: dict[str, set[str]] = field(default_factory=dict)


class RetrievalPipeline:
    """Clear stages for memory retrieval.

    Stage 1: Primary search - Vector search scoped to user/chat
    Stage 2: Cross-context - Facts about participants from other chats
    Stage 3: Graph traversal - Facts about people mentioned in results
    Stage 4: Finalize - Dedupe, rank, limit
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    async def retrieve(self, context: RetrievalContext) -> RetrievedContext:
        """Execute the full retrieval pipeline."""
        # Stage 1: Primary vector search
        memories = await self._primary_search(context)

        # Stage 2: Cross-context retrieval
        if context.participant_person_ids:
            memories.extend(await self._cross_context(context))

        # Stage 3: Graph traversal
        seen_ids = {m.id for m in memories}
        if seen_ids:
            memories.extend(await self._graph_traversal(context, memories, seen_ids))

        # Stage 4: Finalize
        return self._finalize(memories, context.max_memories)

    async def _primary_search(self, context: RetrievalContext) -> list[SearchResult]:
        """Stage 1: Vector search scoped to user/chat."""
        try:
            return await self._store.search(
                query=context.query,
                limit=context.max_memories,
                owner_user_id=context.user_id,
                chat_id=context.chat_id,
            )
        except Exception:
            logger.error(
                "Primary memory search failed, returning without context",
                exc_info=True,
            )
            return []

    async def _cross_context(self, context: RetrievalContext) -> list[SearchResult]:
        """Stage 2: Retrieve memories from other contexts about participants."""
        results: list[SearchResult] = []

        for username, person_ids in context.participant_person_ids.items():
            if not person_ids:
                continue
            try:
                cross_memories = await self._find_memories_about_persons(
                    person_ids=person_ids,
                    exclude_owner_user_id=context.user_id,
                    limit=context.max_memories,
                )
                for memory in cross_memories:
                    if self._passes_privacy_filter(
                        sensitivity=memory.sensitivity,
                        subject_person_ids=memory.subject_person_ids,
                        chat_type=context.chat_type,
                        querying_person_ids=person_ids,
                    ):
                        results.append(
                            await self._make_cross_context_result(
                                memory, CROSS_CONTEXT_SIMILARITY
                            )
                        )
            except Exception:
                logger.warning("Failed to get cross-context memories for %s", username)

        return results

    async def _graph_traversal(
        self,
        context: RetrievalContext,
        memories: list[SearchResult],
        seen_ids: set[str],
    ) -> list[SearchResult]:
        """Stage 3: Find memories about people mentioned in existing results."""
        mentioned_person_ids: set[str] = set()
        for m in memories:
            spids = (m.metadata or {}).get("subject_person_ids") or []
            mentioned_person_ids.update(spids)

        # Exclude participants already handled by cross-context retrieval
        for pids in context.participant_person_ids.values():
            mentioned_person_ids -= pids

        if not mentioned_person_ids:
            return []

        try:
            subject_cross = await self._find_memories_about_persons(
                person_ids=mentioned_person_ids,
                exclude_owner_user_id=context.user_id,
                limit=context.max_memories,
            )
        except Exception:
            logger.warning("Graph traversal cross-context failed", exc_info=True)
            return []

        all_participant_ids: set[str] = set()
        for pids in context.participant_person_ids.values():
            all_participant_ids |= pids

        results: list[SearchResult] = []
        for memory in subject_cross:
            if memory.id in seen_ids:
                continue
            if self._passes_privacy_filter(
                sensitivity=memory.sensitivity,
                subject_person_ids=memory.subject_person_ids,
                chat_type=context.chat_type,
                querying_person_ids=all_participant_ids,
            ):
                results.append(
                    await self._make_cross_context_result(
                        memory,
                        GRAPH_TRAVERSAL_SIMILARITY,
                        graph_traversal=True,
                    )
                )
                seen_ids.add(memory.id)

        return results

    def _finalize(
        self, memories: list[SearchResult], max_memories: int
    ) -> RetrievedContext:
        """Stage 4: Deduplicate, rank, and limit results."""
        unique: list[SearchResult] = []
        deduped: set[str] = set()

        for m in memories:
            if m.id not in deduped:
                deduped.add(m.id)
                unique.append(m)
                if len(unique) >= max_memories:
                    break

        return RetrievedContext(memories=unique)

    async def _find_memories_about_persons(
        self,
        person_ids: set[str],
        exclude_owner_user_id: str | None = None,
        limit: int = 20,
        portable_only: bool = True,
    ) -> list[MemoryEntry]:
        """Find memories about given persons using SQL JOINs."""
        now = datetime.now(UTC)

        result_memories: list[MemoryEntry] = []
        async with self._store._db.session() as session:
            for pid in person_ids:
                r = await session.execute(
                    text("""
                        SELECT m.* FROM memories m
                        JOIN memory_subjects ms ON ms.memory_id = m.id
                        WHERE ms.person_id = :pid
                            AND m.archived_at IS NULL
                            AND m.superseded_at IS NULL
                            AND (m.expires_at IS NULL OR m.expires_at > :now)
                    """),
                    {"pid": pid, "now": now.isoformat()},
                )
                rows = r.fetchall()
                for row in rows:
                    memory = _row_to_memory(row)
                    memory.subject_person_ids = await _load_subjects(session, memory.id)
                    if (
                        exclude_owner_user_id
                        and memory.owner_user_id == exclude_owner_user_id
                    ):
                        continue
                    if portable_only and not memory.portable:
                        continue
                    result_memories.append(memory)
                    if len(result_memories) >= limit:
                        return result_memories

        return result_memories

    def _passes_privacy_filter(
        self,
        sensitivity: Sensitivity | None,
        subject_person_ids: list[str],
        chat_type: str | None,
        querying_person_ids: set[str],
    ) -> bool:
        """Check if a memory passes privacy filter."""
        if sensitivity is None or sensitivity == Sensitivity.PUBLIC:
            return True
        is_subject = bool(set(subject_person_ids) & querying_person_ids)
        if sensitivity == Sensitivity.PERSONAL:
            return is_subject
        if sensitivity == Sensitivity.SENSITIVE:
            return chat_type == "private" and is_subject
        return False

    async def _make_cross_context_result(
        self,
        memory: MemoryEntry,
        similarity: float,
        *,
        graph_traversal: bool = False,
    ) -> SearchResult:
        """Convert a MemoryEntry to a SearchResult for cross-context retrieval."""
        subject_name = await self._store._resolve_subject_name(
            memory.subject_person_ids
        )
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
