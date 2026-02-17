"""Memory retrieval pipeline with multi-hop BFS and RRF fusion.

4-stage pipeline:
  Stage 1: Vector search (scoped to user/chat)
  Stage 2: Cross-context (ABOUT edges for participants)
  Stage 3: Multi-hop BFS (2-hop graph traversal from seed persons)
  Stage 4: RRF fusion (reciprocal rank fusion across stages)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ash.graph.edges import get_memories_about_person
from ash.graph.traversal import bfs_traverse
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

# RRF constant (standard value from the original RRF paper)
RRF_K = 60


@dataclass
class RetrievalContext:
    """Context for a retrieval operation."""

    user_id: str
    query: str
    chat_id: str | None = None
    max_memories: int = 10
    chat_type: str | None = None
    participant_person_ids: dict[str, set[str]] = field(default_factory=dict)
    graph_chat_id: str | None = None  # Graph chat node ID for contextual disclosure


class RetrievalPipeline:
    """Multi-stage memory retrieval with graph traversal and RRF fusion.

    Stage 1: Primary search - Vector search scoped to user/chat
    Stage 2: Cross-context - Facts about participants from other chats
    Stage 3: Multi-hop BFS - 2-hop graph traversal from seed person nodes
    Stage 4: RRF fusion - Reciprocal rank fusion across all stages
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    async def retrieve(self, context: RetrievalContext) -> RetrievedContext:
        """Execute the full retrieval pipeline."""
        # Stage 1: Primary vector search
        stage1 = await self._primary_search(context)
        stage1 = self._filter_results_by_privacy(stage1, context)

        # Stage 2: Cross-context retrieval
        stage2: list[SearchResult] = []
        if context.participant_person_ids:
            stage2 = await self._cross_context(context)

        # Stage 3: Multi-hop BFS traversal
        stage3 = await self._multi_hop_traversal(context, stage1, stage2)

        # Stage 4: RRF fusion
        return self._rrf_finalize([stage1, stage2, stage3], context.max_memories)

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
            logger.error("primary_search_failed", exc_info=True)
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
                    from ash.graph.edges import get_subject_person_ids

                    mem_subjects = get_subject_person_ids(self._store.graph, memory.id)
                    if self._passes_privacy_filter(
                        sensitivity=memory.sensitivity,
                        subject_person_ids=mem_subjects,
                        chat_type=context.chat_type,
                        querying_person_ids=person_ids,
                    ):
                        results.append(
                            await self._make_result(
                                memory,
                                CROSS_CONTEXT_SIMILARITY,
                                discovery_stage="cross_context",
                            )
                        )
            except Exception:
                logger.warning(
                    "cross_context_retrieval_failed",
                    extra={
                        "user.username": username,
                        "person.ids": list(person_ids),
                    },
                )

        return results

    async def _multi_hop_traversal(
        self,
        context: RetrievalContext,
        stage1: list[SearchResult],
        stage2: list[SearchResult],
    ) -> list[SearchResult]:
        """Stage 3: BFS traversal from seed person nodes.

        Seeds are person IDs discovered in stages 1 and 2.
        BFS follows edges up to 2 hops, discovering new memory nodes.
        Applies privacy/scope filtering at each hop.
        """
        graph = self._store.graph

        # Collect seed person IDs from stage 1 and stage 2 results
        seed_person_ids: set[str] = set()
        for m in stage1:
            spids = (m.metadata or {}).get("subject_person_ids") or []
            seed_person_ids.update(spids)
        for m in stage2:
            spids = (m.metadata or {}).get("subject_person_ids") or []
            seed_person_ids.update(spids)

        # Also include participant person IDs
        for pids in context.participant_person_ids.values():
            seed_person_ids |= pids

        if not seed_person_ids:
            return []

        # Collect already-seen memory IDs to avoid duplicates
        seen_ids: set[str] = set()
        for m in stage1:
            seen_ids.add(m.id)
        for m in stage2:
            seen_ids.add(m.id)

        # All participant IDs for privacy filtering
        all_participant_ids: set[str] = set()
        for pids in context.participant_person_ids.values():
            all_participant_ids |= pids

        # BFS from seed persons
        try:
            traversal_results = bfs_traverse(
                graph,
                seed_ids=seed_person_ids,
                max_hops=2,
            )
        except Exception:
            logger.warning(
                "bfs_traversal_failed",
                extra={"person.seed_ids": list(seed_person_ids)},
                exc_info=True,
            )
            return []

        now = datetime.now(UTC)
        results: list[SearchResult] = []

        for tr in traversal_results:
            if tr.node_type != "memory":
                continue
            if tr.node_id in seen_ids:
                continue

            memory = graph.memories.get(tr.node_id)
            if not memory:
                continue
            if memory.archived_at is not None:
                continue
            if memory.superseded_at is not None:
                continue
            if memory.expires_at and memory.expires_at <= now:
                continue
            if not memory.portable:
                continue

            # Privacy filter
            from ash.graph.edges import get_subject_person_ids

            mem_subjects = get_subject_person_ids(graph, memory.id)
            if not self._passes_privacy_filter(
                sensitivity=memory.sensitivity,
                subject_person_ids=mem_subjects,
                chat_type=context.chat_type,
                querying_person_ids=all_participant_ids,
            ):
                continue

            # Score based on hop distance
            hop_similarity = 0.5 if tr.hops == 1 else 0.3
            results.append(
                await self._make_result(
                    memory,
                    hop_similarity,
                    discovery_stage="graph_traversal",
                    hops=tr.hops,
                )
            )
            seen_ids.add(tr.node_id)

        return results

    def _rrf_finalize(
        self,
        stage_results: list[list[SearchResult]],
        max_memories: int,
    ) -> RetrievedContext:
        """Stage 4: RRF fusion across stages, then dedupe and limit.

        RRF score = sum(1 / (k + rank)) across all stages where the memory appears.
        Falls back to similarity-based sort when only one stage has results.
        """
        # Count how many stages have results
        active_stages = [s for s in stage_results if s]

        if not active_stages:
            return RetrievedContext(memories=[])

        if len(active_stages) == 1:
            # Single stage: just sort by similarity (no RRF benefit)
            return self._simple_finalize(active_stages[0], max_memories)

        # Compute RRF scores
        rrf_scores: dict[str, float] = {}
        result_by_id: dict[str, SearchResult] = {}

        for stage in active_stages:
            for rank, result in enumerate(stage):
                rrf_scores[result.id] = rrf_scores.get(result.id, 0.0) + (
                    1.0 / (RRF_K + rank + 1)
                )
                # Keep the highest-similarity version
                existing = result_by_id.get(result.id)
                if existing is None or result.similarity > existing.similarity:
                    result_by_id[result.id] = result

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=lambda mid: rrf_scores[mid], reverse=True)

        unique: list[SearchResult] = []
        for mid in sorted_ids:
            unique.append(result_by_id[mid])
            if len(unique) >= max_memories:
                break

        return RetrievedContext(memories=unique)

    @staticmethod
    def _simple_finalize(
        memories: list[SearchResult], max_memories: int
    ) -> RetrievedContext:
        """Simple dedupe and limit for single-stage results."""
        sorted_memories = sorted(memories, key=lambda m: m.similarity, reverse=True)

        unique: list[SearchResult] = []
        deduped: set[str] = set()

        for m in sorted_memories:
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
        """Find memories about given persons using ABOUT edges."""
        now = datetime.now(UTC)
        graph = self._store.graph

        # Collect candidate memory IDs via ABOUT edges
        candidate_ids: set[str] = set()
        for pid in person_ids:
            candidate_ids.update(get_memories_about_person(graph, pid))

        result_memories: list[MemoryEntry] = []
        for mid in candidate_ids:
            memory = graph.memories.get(mid)
            if not memory:
                continue
            if memory.archived_at is not None:
                continue
            if memory.superseded_at is not None:
                continue
            if memory.expires_at and memory.expires_at <= now:
                continue
            if exclude_owner_user_id and memory.owner_user_id == exclude_owner_user_id:
                continue
            if portable_only and not memory.portable:
                continue

            result_memories.append(memory)

        # Sort by recency so the limit picks the most recent memories
        result_memories.sort(
            key=lambda m: m.created_at or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )
        return result_memories[:limit]

    def _filter_results_by_privacy(
        self,
        results: list[SearchResult],
        context: RetrievalContext,
    ) -> list[SearchResult]:
        """Filter Stage 1 results based on chat context.

        In group chats: filters SENSITIVE and PERSONAL memories about
        non-participants to prevent information leakage.

        In private chats (DMs): applies contextual disclosure — only
        surfaces memories about third parties if the DM partner was
        present when the memory was learned.
        """
        if context.chat_type == "private":
            return self._filter_dm_contextual(results, context)

        if context.chat_type in ("group", "supergroup"):
            return self._filter_group_privacy(results, context)

        return results

    def _filter_dm_contextual(
        self,
        results: list[SearchResult],
        context: RetrievalContext,
    ) -> list[SearchResult]:
        """DM contextual disclosure filter.

        A memory is disclosable if:
        - It's about the DM partner (ABOUT edge)
        - It was stated by the DM partner (STATED_BY edge)
        - The DM partner was in the chat where it was learned (LEARNED_IN → PARTICIPATES_IN)
        - It's a self-memory (no subjects)
        - It has no LEARNED_IN edge (legacy data, fail-open)
        """
        from ash.graph.edges import (
            get_learned_in_chat,
            get_stated_by_person,
            person_participates_in_chat,
        )

        # Collect DM partner person IDs
        partner_person_ids: set[str] = set()
        for pids in context.participant_person_ids.values():
            partner_person_ids |= pids

        if not partner_person_ids:
            return results  # No partner info — fail-open

        graph = self._store.graph
        filtered = []
        for result in results:
            meta = result.metadata or {}
            subject_person_ids = meta.get("subject_person_ids", [])

            # Self-memories (no subjects) always pass
            if not subject_person_ids:
                filtered.append(result)
                continue

            # Memory is ABOUT the DM partner
            if set(subject_person_ids) & partner_person_ids:
                filtered.append(result)
                continue

            # Memory was STATED_BY the DM partner
            stated_by = get_stated_by_person(graph, result.id)
            if stated_by and stated_by in partner_person_ids:
                filtered.append(result)
                continue

            # Check LEARNED_IN → partner was present in that chat
            learned_in_chat = get_learned_in_chat(graph, result.id)
            if learned_in_chat is None:
                # Legacy memory without LEARNED_IN — fail-open
                filtered.append(result)
                continue

            # Partner participated in the source chat
            if any(
                person_participates_in_chat(graph, pid, learned_in_chat)
                for pid in partner_person_ids
            ):
                filtered.append(result)
                continue

            # Memory about a third party, partner not present — exclude

        return filtered

    def _filter_group_privacy(
        self,
        results: list[SearchResult],
        context: RetrievalContext,
    ) -> list[SearchResult]:
        """Group chat privacy filter.

        - PUBLIC: always included
        - PERSONAL about non-participants: excluded
        - PERSONAL about participants: included
        - SENSITIVE about non-participants: excluded
        - SENSITIVE about participants: included
        - Self-memories (no subjects): always included
        """
        all_participant_ids: set[str] = set()
        for pids in context.participant_person_ids.values():
            all_participant_ids |= pids

        filtered = []
        for result in results:
            meta = result.metadata or {}
            subject_person_ids = meta.get("subject_person_ids", [])

            # Self-memories (no subjects) always visible to the owner
            if not subject_person_ids:
                filtered.append(result)
                continue

            sensitivity_str = meta.get("sensitivity")
            if sensitivity_str not in (
                Sensitivity.SENSITIVE.value,
                Sensitivity.PERSONAL.value,
            ):
                filtered.append(result)
                continue

            # SENSITIVE/PERSONAL memory about others: only pass if a subject
            # is among the chat participants
            if set(subject_person_ids) & all_participant_ids:
                filtered.append(result)
            # else: filtered out
        return filtered

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

    async def _make_result(
        self,
        memory: MemoryEntry,
        similarity: float,
        *,
        discovery_stage: str = "primary",
        hops: int = 0,
    ) -> SearchResult:
        """Convert a MemoryEntry to a SearchResult."""
        from ash.graph.edges import get_subject_person_ids
        from ash.store.trust import classify_trust, get_trust_weight

        subject_pids = get_subject_person_ids(self._store.graph, memory.id)
        subject_name = await self._store._resolve_subject_name(subject_pids)

        # Apply trust weight to similarity score
        trust_level = classify_trust(self._store.graph, memory.id)
        trust_weight = get_trust_weight(trust_level)
        weighted_similarity = similarity * trust_weight

        meta: dict[str, Any] = {
            "memory_type": memory.memory_type.value,
            "subject_person_ids": subject_pids,
            "discovery_stage": discovery_stage,
            "trust": trust_level,
        }
        if hops > 0:
            meta["hops"] = hops
            meta["graph_traversal"] = True
        if discovery_stage == "cross_context":
            meta["cross_context"] = True
        if subject_name:
            meta["subject_name"] = subject_name
        return SearchResult(
            id=memory.id,
            content=memory.content,
            similarity=weighted_similarity,
            metadata=meta,
            source_type="memory",
        )
