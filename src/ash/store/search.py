"""Search and context retrieval mixin for Store (SQLite-backed)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from ash.store.mappers import row_to_memory as _row_to_memory
from ash.store.memories.helpers import load_subjects as _load_subjects
from ash.store.retrieval import RetrievalContext, RetrievalPipeline
from ash.store.types import (
    MemoryEntry,
    RetrievedContext,
    SearchResult,
    Sensitivity,
    matches_scope,
)

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)

# Default similarity scores for non-vector results (used for ranking only)
_CROSS_CONTEXT_SIMILARITY = 0.7
_GRAPH_TRAVERSAL_SIMILARITY = 0.6


class SearchMixin:
    """Search, context retrieval, and privacy filtering."""

    async def search(
        self: Store,
        query: str,
        limit: int = 5,
        subject_person_id: str | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[SearchResult]:
        vector_results = await self._index.search(query, limit=limit * 2)
        now = datetime.now(UTC)

        results: list[SearchResult] = []
        for vr in vector_results:
            async with self._db.session() as session:
                r = await session.execute(
                    text("SELECT * FROM memories WHERE id = :id"),
                    {"id": vr.memory_id},
                )
                row = r.fetchone()
                if not row:
                    continue
                memory = _row_to_memory(row)
                memory.subject_person_ids = await _load_subjects(session, memory.id)

            if memory.superseded_at:
                continue
            if memory.archived_at is not None:
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
        self: Store,
        person_ids: set[str],
        exclude_owner_user_id: str | None = None,
        limit: int = 20,
        portable_only: bool = True,
    ) -> list[MemoryEntry]:
        """Find memories about given persons using SQL JOINs."""
        now = datetime.now(UTC)

        result_memories: list[MemoryEntry] = []
        async with self._db.session() as session:
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
        self: Store,
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
        self: Store,
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
        self: Store,
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
        self: Store,
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
