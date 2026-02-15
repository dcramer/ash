"""Supersession and conflict detection mixin for GraphStore."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ash.memory.types import MemoryEntry, matches_scope

if TYPE_CHECKING:
    from ash.graph.store import GraphStore

logger = logging.getLogger(__name__)

CONFLICT_SIMILARITY_THRESHOLD = 0.75

SUPERSESSION_PROMPT = """Given these two memories, determine if the NEW memory supersedes/replaces the OLD memory.

OLD: "{old_content}"
NEW: "{new_content}"

Answer YES if the new memory updates, corrects, or replaces the old memory.
Answer NO if they are about different things or both should be kept.

Answer only YES or NO."""


class SupersessionMixin:
    """Conflict detection, supersession, and hearsay management."""

    async def find_conflicting_memories(
        self: GraphStore,
        new_content: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        subject_person_ids: list[str] | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        similar = await self._index.search(new_content, limit=10)
        now = datetime.now(UTC)

        await self._ensure_graph_built()

        conflicts: list[tuple[MemoryEntry, float]] = []
        for result in similar:
            if result.similarity < CONFLICT_SIMILARITY_THRESHOLD:
                continue
            memory = self._memory_by_id.get(result.memory_id)
            if not memory:
                continue
            if memory.superseded_at:
                continue
            if memory.expires_at and memory.expires_at <= now:
                continue
            if not matches_scope(memory, owner_user_id, chat_id):
                continue
            memory_subjects = memory.subject_person_ids or []
            if subject_person_ids:
                if not set(subject_person_ids) & set(memory_subjects):
                    continue
            elif memory_subjects:
                continue
            conflicts.append((memory, result.similarity))

        return conflicts

    async def _verify_conflict_with_llm(
        self: GraphStore,
        old_content: str,
        new_content: str,
    ) -> bool:
        if not self._llm:
            return False
        try:
            from ash.llm.types import Message, Role

            prompt = SUPERSESSION_PROMPT.format(
                old_content=old_content, new_content=new_content
            )
            response = await self._llm.complete(
                messages=[Message(role=Role.USER, content=prompt)],
                model=self._llm_model,
                max_tokens=10,
                temperature=0.0,
            )
            answer = response.message.get_text().strip().upper()
            return answer.startswith("YES")
        except Exception:
            logger.warning(
                "LLM verification failed, skipping supersession", exc_info=True
            )
            return False

    async def supersede_conflicting_memories(
        self: GraphStore,
        new_memory: MemoryEntry,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> int:
        conflicts = await self.find_conflicting_memories(
            new_content=new_memory.content,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            subject_person_ids=new_memory.subject_person_ids,
        )

        count = 0
        for memory, similarity in conflicts:
            if memory.id == new_memory.id:
                continue
            if await self._is_protected_by_subject_authority(memory, new_memory):
                continue
            if similarity < 0.85 and self._llm:
                if not await self._verify_conflict_with_llm(
                    old_content=memory.content, new_content=new_memory.content
                ):
                    continue
            if await self._mark_superseded(
                memory.id, new_memory.id, invalidate_graph=False
            ):
                count += 1

        if count > 0:
            self._graph_built = False

        return count

    async def _is_protected_by_subject_authority(
        self: GraphStore,
        candidate: MemoryEntry,
        new_memory: MemoryEntry,
    ) -> bool:
        if not candidate.source_username or not candidate.subject_person_ids:
            return False
        if not new_memory.source_username:
            return False
        if new_memory.source_username == candidate.source_username:
            return False

        try:
            source_ids = await self.find_person_ids_for_username(
                candidate.source_username
            )
            if not (source_ids & set(candidate.subject_person_ids)):
                return False
            new_source_ids = await self.find_person_ids_for_username(
                new_memory.source_username
            )
            if new_source_ids & set(candidate.subject_person_ids):
                return False
            return True
        except Exception:
            logger.error(
                "Subject authority check failed, conservatively protecting memory",
                extra={
                    "candidate_id": candidate.id,
                    "new_memory_id": new_memory.id,
                    "candidate_source": candidate.source_username,
                },
                exc_info=True,
            )
            return True

    async def _mark_superseded(
        self: GraphStore,
        old_memory_id: str,
        new_memory_id: str,
        *,
        invalidate_graph: bool = True,
    ) -> bool:
        success = await self._store.mark_memory_superseded(
            memory_id=old_memory_id, superseded_by_id=new_memory_id
        )
        if success:
            # Keep _memory_by_id in sync so lookups see the superseded state
            old_memory = self._memory_by_id.get(old_memory_id)
            if old_memory:
                old_memory.superseded_at = datetime.now(UTC)
                old_memory.superseded_by_id = new_memory_id
            try:
                await self._index.delete_embedding(old_memory_id)
            except Exception:
                logger.warning(
                    "Failed to delete superseded memory embedding",
                    extra={"memory_id": old_memory_id},
                    exc_info=True,
                )
            if invalidate_graph:
                self._graph_built = False
        return success

    async def mark_superseded(
        self: GraphStore, old_memory_id: str, new_memory_id: str
    ) -> bool:
        """Mark a memory as superseded (public API for doctor/dedup)."""
        return await self._mark_superseded(old_memory_id, new_memory_id)

    async def batch_mark_superseded(
        self: GraphStore, pairs: list[tuple[str, str]]
    ) -> list[str]:
        """Mark multiple memories as superseded in a single batch.

        Args:
            pairs: List of (old_memory_id, new_memory_id) tuples.

        Returns:
            List of old memory IDs that were actually marked.
        """
        if not pairs:
            return []

        marked = await self._store.batch_mark_superseded(pairs)
        if marked:
            try:
                await self._index.delete_embeddings(marked)
            except Exception:
                logger.warning(
                    "Failed to delete embeddings for superseded memories",
                    extra={"count": len(marked)},
                    exc_info=True,
                )
            self._graph_built = False
        return marked

    async def supersede_confirmed_hearsay(
        self: GraphStore,
        new_memory: MemoryEntry,
        person_ids: set[str],
        source_username: str,
        owner_user_id: str,
        similarity_threshold: float = 0.80,
    ) -> int:
        graph = await self._ensure_graph_built()
        candidate_ids: set[str] = set()
        for pid in person_ids:
            candidate_ids |= graph.memories_about(pid)

        user_lower = source_username.lower()
        hearsay_candidates: list[MemoryEntry] = []
        for mid in candidate_ids:
            memory = self._memory_by_id.get(mid)
            if not memory:
                continue
            if memory.archived_at is not None:
                continue
            if memory.superseded_at:
                continue
            if memory.expires_at and memory.expires_at <= datetime.now(UTC):
                continue
            if owner_user_id and memory.owner_user_id != owner_user_id:
                continue
            # Skip facts (user speaking about themselves)
            if (memory.source_username or "").lower() == user_lower:
                continue
            hearsay_candidates.append(memory)

        if not hearsay_candidates:
            return 0

        # Single vector search instead of N per-candidate searches
        try:
            similar = await self._index.search(
                new_memory.content, limit=len(hearsay_candidates) + 5
            )
        except Exception:
            logger.warning("Failed to search for hearsay similarity", exc_info=True)
            return 0

        similarity_by_id: dict[str, float] = {
            r.memory_id: r.similarity for r in similar
        }

        count = 0
        for hearsay in hearsay_candidates:
            if hearsay.id == new_memory.id:
                continue
            similarity = similarity_by_id.get(hearsay.id, 0.0)
            if similarity < similarity_threshold:
                continue
            try:
                if similarity < 0.85 and self._llm:
                    if not await self._verify_conflict_with_llm(
                        old_content=hearsay.content,
                        new_content=new_memory.content,
                    ):
                        continue
                if await self._mark_superseded(
                    hearsay.id, new_memory.id, invalidate_graph=False
                ):
                    count += 1
                    logger.info(
                        "Hearsay superseded by fact",
                        extra={
                            "hearsay_id": hearsay.id,
                            "fact_id": new_memory.id,
                            "similarity": similarity,
                        },
                    )
            except Exception:
                logger.warning(
                    "Failed to check hearsay similarity",
                    extra={"hearsay_id": hearsay.id},
                    exc_info=True,
                )

        if count > 0:
            self._graph_built = False

        return count
