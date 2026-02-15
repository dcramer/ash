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
        vector_results = await self._index.search(query, limit=limit * 2)
        now = datetime.now(UTC)

        results: list[SearchResult] = []
        async with self._db.session() as session:
            for vr in vector_results:
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

                subject_name = await self._resolve_subject_name(
                    memory.subject_person_ids
                )
                metadata: dict[str, Any] = {
                    "memory_type": memory.memory_type.value,
                    "subject_person_ids": memory.subject_person_ids,
                    "sensitivity": memory.sensitivity.value
                    if memory.sensitivity
                    else None,
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
