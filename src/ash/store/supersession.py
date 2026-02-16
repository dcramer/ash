"""Supersession and conflict detection mixin for Store (in-memory graph backed)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ash.graph.edges import (
    create_supersedes_edge,
    get_subject_person_ids,
)
from ash.store.types import MemoryEntry, matches_scope

if TYPE_CHECKING:
    from ash.store.store import Store

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

    async def _resolve_person_ids_through_merges(
        self: Store, person_ids: list[str]
    ) -> set[str]:
        """Resolve person IDs through merge chains to canonical IDs."""
        from ash.graph.edges import get_merged_into

        resolved: set[str] = set()
        for pid in person_ids:
            person = await self.get_person(pid)
            if person and get_merged_into(self._graph, person.id):
                primary = await self._follow_merge_chain(person)
                resolved.add(primary.id)
            elif person:
                resolved.add(person.id)
            else:
                resolved.add(pid)
        return resolved

    async def find_conflicting_memories(
        self: Store,
        new_content: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        subject_person_ids: list[str] | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        try:
            query_embedding = await self._embeddings.embed(new_content)
        except Exception:
            return []

        similar = self._index.search(query_embedding, limit=10)
        now = datetime.now(UTC)

        conflicts: list[tuple[MemoryEntry, float]] = []
        for memory_id, similarity in similar:
            if similarity < CONFLICT_SIMILARITY_THRESHOLD:
                continue
            memory = self._graph.memories.get(memory_id)
            if not memory:
                continue
            if memory.superseded_at:
                continue
            if memory.expires_at and memory.expires_at <= now:
                continue
            if not matches_scope(memory, owner_user_id, chat_id):
                continue
            memory_subjects = get_subject_person_ids(self._graph, memory.id)
            if subject_person_ids:
                resolved_new = await self._resolve_person_ids_through_merges(
                    subject_person_ids
                )
                resolved_mem = await self._resolve_person_ids_through_merges(
                    memory_subjects
                )
                if not resolved_new & resolved_mem:
                    continue
            elif memory_subjects:
                continue
            conflicts.append((memory, similarity))

        return conflicts

    async def _verify_conflict_with_llm(
        self: Store,
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
        self: Store,
        new_memory: MemoryEntry,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> int:
        new_subjects = get_subject_person_ids(self._graph, new_memory.id)
        conflicts = await self.find_conflicting_memories(
            new_content=new_memory.content,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            subject_person_ids=new_subjects or None,
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
            if await self._mark_superseded(memory.id, new_memory.id):
                count += 1

        return count

    async def _is_protected_by_subject_authority(
        self: Store,
        candidate: MemoryEntry,
        new_memory: MemoryEntry,
    ) -> bool:
        candidate_subjects = get_subject_person_ids(self._graph, candidate.id)
        if not candidate.source_username or not candidate_subjects:
            return False
        if not new_memory.source_username:
            return False
        if new_memory.source_username == candidate.source_username:
            return False

        try:
            source_ids = await self.find_person_ids_for_username(
                candidate.source_username
            )
            if not (source_ids & set(candidate_subjects)):
                return False
            new_source_ids = await self.find_person_ids_for_username(
                new_memory.source_username
            )
            if new_source_ids & set(candidate_subjects):
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
        self: Store,
        old_memory_id: str,
        new_memory_id: str,
    ) -> bool:
        memory = self._graph.memories.get(old_memory_id)
        if not memory or memory.superseded_at is not None:
            return False

        memory.superseded_at = datetime.now(UTC)
        self._graph.add_edge(create_supersedes_edge(new_memory_id, old_memory_id))
        self._persistence.mark_dirty("memories", "edges")
        await self._persistence.flush(self._graph)

        try:
            self._index.remove(old_memory_id)
            await self._index.save(
                self._persistence.graph_dir / "embeddings" / "memories.npy"
            )
        except Exception:
            logger.warning(
                "Failed to delete superseded memory embedding",
                extra={"memory_id": old_memory_id},
                exc_info=True,
            )
        return True

    def _mark_superseded_batched(
        self: Store,
        old_memory_id: str,
        new_memory_id: str,
    ) -> bool:
        """Mark superseded in-memory and mark dirty. Caller must flush."""
        memory = self._graph.memories.get(old_memory_id)
        if not memory or memory.superseded_at is not None:
            return False

        memory.superseded_at = datetime.now(UTC)
        self._graph.add_edge(create_supersedes_edge(new_memory_id, old_memory_id))
        self._persistence.mark_dirty("memories", "edges")

        try:
            self._index.remove(old_memory_id)
        except Exception:
            logger.warning(
                "Failed to delete superseded memory embedding",
                extra={"memory_id": old_memory_id},
                exc_info=True,
            )
        return True

    async def _supersede_conflicting_batched(
        self: Store,
        new_memory: MemoryEntry,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> int:
        """Like supersede_conflicting_memories but marks dirty instead of saving."""
        new_subjects = get_subject_person_ids(self._graph, new_memory.id)
        conflicts = await self.find_conflicting_memories(
            new_content=new_memory.content,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            subject_person_ids=new_subjects or None,
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
            if self._mark_superseded_batched(memory.id, new_memory.id):
                count += 1

        return count

    async def mark_superseded(
        self: Store, old_memory_id: str, new_memory_id: str
    ) -> bool:
        """Mark a memory as superseded (public API for doctor/dedup)."""
        return await self._mark_superseded(old_memory_id, new_memory_id)

    async def batch_mark_superseded(
        self: Store, pairs: list[tuple[str, str]]
    ) -> list[str]:
        """Mark multiple memories as superseded in a single batch."""
        if not pairs:
            return []

        now = datetime.now(UTC)
        marked: list[str] = []

        for old_id, new_id in pairs:
            memory = self._graph.memories.get(old_id)
            if memory and memory.superseded_at is None:
                memory.superseded_at = now
                self._graph.add_edge(create_supersedes_edge(new_id, old_id))
                marked.append(old_id)

        if marked:
            self._persistence.mark_dirty("memories", "edges")
            await self._persistence.flush(self._graph)

            for mid in marked:
                try:
                    self._index.remove(mid)
                except Exception:
                    logger.debug("Failed to remove embedding for %s", mid)

            try:
                await self._index.save(
                    self._persistence.graph_dir / "embeddings" / "memories.npy"
                )
            except Exception:
                logger.warning(
                    "Failed to delete embeddings for superseded memories",
                    extra={"count": len(marked)},
                    exc_info=True,
                )

        return marked

    async def supersede_confirmed_hearsay(
        self: Store,
        new_memory: MemoryEntry,
        person_ids: set[str],
        source_username: str,
        similarity_threshold: float = 0.80,
    ) -> int:
        now = datetime.now(UTC)
        user_lower = source_username.lower()

        # Resolve person IDs through merge chains
        resolved_person_ids = await self._resolve_person_ids_through_merges(
            list(person_ids)
        )
        all_search_ids = person_ids | resolved_person_ids

        # Find candidate hearsay memories about these persons via ABOUT edges
        from ash.graph.edges import get_memories_about_person
        from ash.store.trust import classify_trust

        candidate_mids: set[str] = set()
        for pid in all_search_ids:
            candidate_mids.update(get_memories_about_person(self._graph, pid))

        hearsay_candidates: list[MemoryEntry] = []
        for mid in candidate_mids:
            memory = self._graph.memories.get(mid)
            if not memory:
                continue
            if memory.archived_at is not None:
                continue
            if memory.superseded_at is not None:
                continue
            if memory.expires_at and memory.expires_at <= now:
                continue
            # Use trust classification: skip facts (speaker is subject)
            trust = classify_trust(self._graph, mid)
            if trust == "fact":
                continue
            # Fallback for memories without STATED_BY edges: use source_username
            if (
                trust == "unknown"
                and (memory.source_username or "").lower() == user_lower
            ):
                continue
            hearsay_candidates.append(memory)

        if not hearsay_candidates:
            return 0

        # Deduplicate by ID
        seen_ids: set[str] = set()
        unique_candidates: list[MemoryEntry] = []
        for mem in hearsay_candidates:
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                unique_candidates.append(mem)
        hearsay_candidates = unique_candidates

        # Single vector search
        try:
            query_embedding = await self._embeddings.embed(new_memory.content)
            similar = self._index.search(
                query_embedding, limit=len(hearsay_candidates) + 5
            )
        except Exception:
            logger.warning("Failed to search for hearsay similarity", exc_info=True)
            return 0

        similarity_by_id: dict[str, float] = {mid: sim for mid, sim in similar}

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
                if await self._mark_superseded(hearsay.id, new_memory.id):
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

        return count
