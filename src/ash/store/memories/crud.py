"""Memory CRUD operations: create, read, update, delete."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ash.graph.edges import create_about_edge
from ash.memory.secrets import contains_secret
from ash.store.types import (
    AssertionEnvelope,
    AssertionKind,
    AssertionPredicate,
    MemoryEntry,
    MemoryType,
    PredicateObjectType,
    Sensitivity,
    matches_scope,
    upsert_assertion_metadata,
)

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)


class MemoryCrudMixin:
    """Memory create, read, update, delete operations."""

    async def _normalize_semantics_for_write(
        self: Store,
        *,
        content: str,
        memory_type: MemoryType,
        metadata: dict[str, Any] | None,
        subject_person_ids: list[str],
        stated_by_person_id: str | None,
        source_username: str | None,
    ) -> tuple[list[str], str | None, dict[str, Any] | None]:
        """Normalize assertion + ABOUT/STATED_BY semantics at write-time."""
        subject_pids = self._canonicalize_person_ids(subject_person_ids)
        stated_by = self._canonicalize_person_id(stated_by_person_id)

        assertion: AssertionEnvelope | None = None
        raw_assertion = (metadata or {}).get("assertion") if metadata else None
        if isinstance(raw_assertion, dict):
            try:
                assertion = AssertionEnvelope.model_validate(raw_assertion)
            except Exception:
                assertion = None

        if assertion is not None:
            assertion = assertion.model_copy(
                update={
                    "subjects": self._canonicalize_person_ids(assertion.subjects),
                    "speaker_person_id": self._canonicalize_person_id(
                        assertion.speaker_person_id
                    ),
                }
            )
            if not subject_pids and assertion.subjects:
                subject_pids = list(assertion.subjects)
            if subject_pids and not assertion.subjects:
                assertion.subjects = list(subject_pids)
            if stated_by is None and assertion.speaker_person_id is not None:
                stated_by = assertion.speaker_person_id
            if stated_by is not None and assertion.speaker_person_id is None:
                assertion.speaker_person_id = stated_by

        # Trust parity: infer STATED_BY from source username when resolvable.
        if stated_by is None and source_username:
            try:
                source_person_ids = await self.find_person_ids_for_username(
                    source_username
                )
            except Exception:
                source_person_ids = set()
            if source_person_ids:
                stated_by = sorted(source_person_ids)[0]
                if assertion is not None and assertion.speaker_person_id is None:
                    assertion.speaker_person_id = stated_by

        # Ingestion parity: self-facts are linked to speaker person when available.
        if not subject_pids and stated_by and memory_type != MemoryType.RELATIONSHIP:
            subject_pids = [stated_by]
            if assertion is not None and not assertion.subjects:
                assertion.subjects = [stated_by]

        if (
            assertion is not None
            and assertion.subjects
            and assertion.assertion_kind
            in {
                AssertionKind.CONTEXT_FACT,
                AssertionKind.GROUP_FACT,
            }
        ):
            if memory_type == MemoryType.RELATIONSHIP:
                assertion.assertion_kind = AssertionKind.RELATIONSHIP_FACT
            elif stated_by and set(assertion.subjects) == {stated_by}:
                assertion.assertion_kind = AssertionKind.SELF_FACT
            else:
                assertion.assertion_kind = AssertionKind.PERSON_FACT

        if assertion is None and (
            subject_pids or stated_by or memory_type == MemoryType.RELATIONSHIP
        ):
            if memory_type == MemoryType.RELATIONSHIP:
                assertion_kind = AssertionKind.RELATIONSHIP_FACT
            elif subject_pids and stated_by and set(subject_pids) == {stated_by}:
                assertion_kind = AssertionKind.SELF_FACT
            elif subject_pids:
                assertion_kind = AssertionKind.PERSON_FACT
            else:
                assertion_kind = AssertionKind.CONTEXT_FACT

            assertion = AssertionEnvelope(
                assertion_kind=assertion_kind,
                subjects=subject_pids,
                speaker_person_id=stated_by,
                predicates=[
                    AssertionPredicate(
                        name="describes",
                        object_type=PredicateObjectType.TEXT,
                        value=content,
                    )
                ],
            )

        if assertion is not None:
            metadata = upsert_assertion_metadata(metadata, assertion)

        return subject_pids, stated_by, metadata

    async def _resolve_subject_name(self: Store, person_ids: list[str]) -> str | None:
        if not person_ids:
            return None

        names: list[str] = []
        cache_misses: list[str] = []

        for pid in person_ids:
            if pid in self._person_name_cache:
                names.append(self._person_name_cache[pid])
            else:
                cache_misses.append(pid)

        if cache_misses:
            batch_names: dict[str, str] = {}
            try:
                batch_names = await self.get_person_names_batch(cache_misses)
            except Exception:
                logger.debug("Failed to batch resolve person names, falling back")

            for pid in cache_misses:
                if pid in batch_names:
                    name = batch_names[pid]
                    self._person_name_cache[pid] = name
                    names.append(name)
                else:
                    try:
                        person = await self.get_person(pid)
                        if person:
                            self._person_name_cache[pid] = person.name
                            names.append(person.name)
                    except Exception:
                        logger.debug(
                            "person_name_resolve_failed",
                            extra={"person.id": pid},
                        )

        return ", ".join(names) if names else None

    async def add_memory(
        self: Store,
        content: str,
        source: str = "user",
        memory_type: MemoryType | None = None,
        expires_at: datetime | None = None,
        expires_in_days: int | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        subject_person_ids: list[str] | None = None,
        observed_at: datetime | None = None,
        source_username: str | None = None,
        source_display_name: str | None = None,
        source_session_id: str | None = None,
        source_message_id: str | None = None,
        extraction_confidence: float | None = None,
        sensitivity: Sensitivity | None = None,
        portable: bool = True,
        metadata: dict[str, Any] | None = None,
        stated_by_person_id: str | None = None,
        graph_chat_id: str | None = None,
        assertion: AssertionEnvelope | None = None,
    ) -> MemoryEntry:
        """Add a memory entry."""
        if contains_secret(content):
            raise ValueError(
                "Memory content contains potential secrets and cannot be stored"
            )

        if expires_in_days is not None and expires_at is None:
            expires_at = datetime.now(UTC) + timedelta(days=expires_in_days)

        if memory_type is None:
            memory_type = MemoryType.KNOWLEDGE

        embedding_floats: list[float] | None = None
        try:
            embedding_floats = await self._embeddings.embed(content)
        except Exception:
            logger.warning("embedding_generation_failed", exc_info=True)

        now = datetime.now(UTC)
        memory_id = str(uuid.uuid4())
        subject_pids = self._canonicalize_person_ids(subject_person_ids or [])
        canonical_stated_by = self._canonicalize_person_id(stated_by_person_id)

        if assertion is not None:
            if assertion.subjects:
                subject_pids = self._canonicalize_person_ids(assertion.subjects)
            if assertion.speaker_person_id:
                canonical_stated_by = self._canonicalize_person_id(
                    assertion.speaker_person_id
                )

            assertion = assertion.model_copy(
                update={
                    "subjects": subject_pids,
                    "speaker_person_id": canonical_stated_by,
                }
            )
            metadata = upsert_assertion_metadata(metadata, assertion)

        (
            subject_pids,
            canonical_stated_by,
            metadata,
        ) = await self._normalize_semantics_for_write(
            content=content,
            memory_type=memory_type,
            metadata=metadata,
            subject_person_ids=subject_pids,
            stated_by_person_id=canonical_stated_by,
            source_username=source_username,
        )

        memory = MemoryEntry(
            id=memory_id,
            version=1,
            content=content,
            memory_type=memory_type,
            created_at=now,
            observed_at=observed_at,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            source=source,
            source_username=source_username,
            source_display_name=source_display_name,
            source_session_id=source_session_id,
            source_message_id=source_message_id,
            extraction_confidence=extraction_confidence,
            sensitivity=sensitivity or Sensitivity.PUBLIC,
            portable=portable,
            expires_at=expires_at,
            metadata=metadata,
        )

        # Add to graph (in-memory)
        self._graph.add_memory(memory)
        self._persistence.mark_dirty("memories")

        # Create ABOUT edges for subject_person_ids
        if subject_pids:
            for pid in subject_pids:
                edge = create_about_edge(memory_id, pid, created_by=source)
                self._graph.add_edge(edge)
            self._persistence.mark_dirty("edges")

        # Create STATED_BY edge for speaker attribution
        if canonical_stated_by:
            from ash.graph.edges import create_stated_by_edge

            self._graph.add_edge(
                create_stated_by_edge(memory_id, canonical_stated_by, created_by=source)
            )
            self._persistence.mark_dirty("edges")

        # Create LEARNED_IN edge to track source chat
        logger.debug(
            "add_memory_graph_chat_id: memory=%s graph_chat_id=%s",
            memory_id[:8],
            graph_chat_id,
        )
        if graph_chat_id:
            from ash.graph.edges import create_learned_in_edge

            edge = create_learned_in_edge(memory_id, graph_chat_id, created_by=source)
            self._graph.add_edge(edge)
            self._persistence.mark_dirty("edges")
            logger.debug(
                "learned_in_edge_created: memory=%s chat=%s edge=%s",
                memory_id[:8],
                graph_chat_id[:8],
                edge.id[:8],
            )

        if embedding_floats:
            embedding_base64 = MemoryEntry.encode_embedding(embedding_floats)
            try:
                self._index.add(memory.id, embedding_floats)
            except Exception:
                logger.warning(
                    "memory_index_failed",
                    extra={"memory.id": memory.id},
                    exc_info=True,
                )
            memory.embedding = embedding_base64

        superseded_count = 0
        try:
            superseded_count = await self._supersede_conflicting_batched(
                new_memory=memory,
                owner_user_id=owner_user_id,
                chat_id=chat_id,
            )
            if superseded_count > 0:
                logger.info(
                    "memory_superseded",
                    extra={
                        "memory.id": memory.id,
                        "superseded_count": superseded_count,
                    },
                )
        except Exception:
            logger.warning(
                "conflicting_memories_check_failed",
                extra={"memory.id": memory.id},
                exc_info=True,
            )

        if self._max_entries is not None:
            try:
                evicted = await self.enforce_max_entries(self._max_entries)
                if evicted > 0:
                    logger.info(
                        "memories_evicted",
                        extra={"count": evicted, "max_entries": self._max_entries},
                    )
            except Exception:
                logger.warning("max_entries_enforcement_failed", exc_info=True)

        # Single flush for all mutations in this operation
        await self._persistence.flush(self._graph)

        # Save vector index after flush (separate file)
        if embedding_floats or superseded_count > 0:
            await self._save_vector_index()

        logger.debug(
            "memory_added",
            extra={
                "memory_id": memory.id,
                "memory_type": memory_type.value,
                "source": source,
            },
        )

        return memory

    def _canonicalize_person_ids(self: Store, person_ids: list[str]) -> list[str]:
        """Canonicalize person IDs through merge chains and dedupe."""
        from ash.graph.edges import follow_merge_chain

        canonical_ids: list[str] = []
        seen: set[str] = set()
        for pid in person_ids:
            if not pid:
                continue
            canonical = pid
            if pid in self._graph.people:
                canonical = follow_merge_chain(self._graph, pid)
            if canonical in seen:
                continue
            seen.add(canonical)
            canonical_ids.append(canonical)
        return canonical_ids

    def _canonicalize_person_id(self: Store, person_id: str | None) -> str | None:
        if not person_id:
            return None
        canonical = self._canonicalize_person_ids([person_id])
        return canonical[0] if canonical else None

    async def get_memory(self: Store, memory_id: str) -> MemoryEntry | None:
        memory = self._graph.memories.get(memory_id)
        if memory and memory.archived_at is None:
            return memory.model_copy(deep=True)
        return None

    async def get_memory_by_prefix(self: Store, prefix: str) -> MemoryEntry | None:
        """Find a memory by ID prefix match."""
        # Try exact match first
        memory = self._graph.memories.get(prefix)
        if memory and memory.archived_at is None:
            return memory.model_copy(deep=True)

        # Try prefix match
        matches = [
            m
            for mid, m in self._graph.memories.items()
            if mid.startswith(prefix) and m.archived_at is None
        ]
        if len(matches) == 1:
            return matches[0].model_copy(deep=True)
        return None

    async def list_memories(
        self: Store,
        limit: int | None = 20,
        include_expired: bool = False,
        include_superseded: bool = False,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        learned_in_chat_id: str | None = None,
    ) -> list[MemoryEntry]:
        from ash.graph.edges import get_memories_learned_in_chat

        # Pre-compute learned-in filter set
        learned_in_ids: set[str] | None = None
        if learned_in_chat_id:
            learned_in_ids = get_memories_learned_in_chat(
                self._graph, learned_in_chat_id
            )

        now = datetime.now(UTC)
        results: list[MemoryEntry] = []

        for memory in self._graph.memories.values():
            if memory.archived_at is not None:
                continue
            if not include_expired:
                if memory.expires_at and memory.expires_at <= now:
                    continue
            if not include_superseded:
                if memory.superseded_at:
                    continue
            if not matches_scope(memory, owner_user_id, chat_id):
                continue
            if learned_in_ids is not None and memory.id not in learned_in_ids:
                continue
            results.append(memory)

        # Sort by created_at descending
        results.sort(
            key=lambda m: m.created_at or datetime.min.replace(tzinfo=UTC), reverse=True
        )

        if limit is not None:
            results = results[:limit]

        return [memory.model_copy(deep=True) for memory in results]

    async def delete_memory(
        self: Store,
        memory_id: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> bool:
        memory = await self.get_memory_by_prefix(memory_id)
        if not memory:
            return False

        full_id = memory.id

        # Check authorization
        if owner_user_id or chat_id:
            is_owner = owner_user_id and memory.owner_user_id == owner_user_id
            is_group_member = memory.owner_user_id is None and memory.chat_id == chat_id
            if not (is_owner or is_group_member):
                return False

        await self.archive_memories({full_id}, reason="user_deleted")

        logger.info("memory_deleted", extra={"memory.id": full_id})
        return True

    async def batch_update_memories(
        self: Store,
        entries: list[MemoryEntry],
        subject_person_ids_map: dict[str, list[str]] | None = None,
    ) -> int:
        """Update multiple memories.

        Args:
            entries: Memory entries to update.
            subject_person_ids_map: Optional map of memory_id -> desired subject person IDs.
                When omitted, current ABOUT edges are used as the baseline and
                normalized semantics may still update them.
        """
        if not entries:
            return 0

        count = 0
        edges_changed = False
        vector_index_changed = False
        for entry in entries:
            if entry.id in self._graph.memories:
                from ash.graph.edges import ABOUT, STATED_BY, create_stated_by_edge

                existing = self._graph.memories[entry.id]
                content_changed = existing.content != entry.content

                existing_about = self._graph.get_outgoing(entry.id, edge_type=ABOUT)
                existing_subjects = [e.target_id for e in existing_about]
                desired_subjects = (
                    subject_person_ids_map.get(entry.id, existing_subjects)
                    if subject_person_ids_map
                    else existing_subjects
                )

                existing_stated_edges = self._graph.get_outgoing(
                    entry.id, edge_type=STATED_BY
                )
                existing_stated = (
                    existing_stated_edges[0].target_id
                    if existing_stated_edges
                    else None
                )

                (
                    desired_subjects,
                    desired_stated_by,
                    normalized_metadata,
                ) = await self._normalize_semantics_for_write(
                    content=entry.content,
                    memory_type=entry.memory_type,
                    metadata=entry.metadata,
                    subject_person_ids=desired_subjects,
                    stated_by_person_id=existing_stated,
                    source_username=entry.source_username,
                )

                entry.metadata = normalized_metadata
                self._graph.memories[entry.id] = entry
                count += 1

                if content_changed:
                    # Keep vector index synchronized with rewritten content.
                    try:
                        self._index.remove(entry.id)
                    except Exception:
                        logger.warning(
                            "memory_index_remove_failed",
                            extra={"memory.id": entry.id},
                            exc_info=True,
                        )

                    entry.embedding = ""
                    try:
                        embedding_floats = await self._embeddings.embed(entry.content)
                    except Exception:
                        logger.warning(
                            "embedding_generation_failed_on_update",
                            extra={"memory.id": entry.id},
                            exc_info=True,
                        )
                        embedding_floats = None

                    if embedding_floats:
                        try:
                            self._index.add(entry.id, embedding_floats)
                            entry.embedding = MemoryEntry.encode_embedding(
                                embedding_floats
                            )
                        except Exception:
                            logger.warning(
                                "memory_index_add_failed_on_update",
                                extra={"memory.id": entry.id},
                                exc_info=True,
                            )
                    vector_index_changed = True

                # Always sync ABOUT edges to normalized subject attribution.
                existing_pids = {e.target_id for e in existing_about}
                desired_pids = set(desired_subjects)

                for edge in existing_about:
                    if edge.target_id not in desired_pids:
                        self._graph.remove_edge(edge.id)
                        edges_changed = True

                for pid in desired_pids - existing_pids:
                    edge = create_about_edge(entry.id, pid, created_by=entry.source)
                    self._graph.add_edge(edge)
                    edges_changed = True

                # Sync STATED_BY edges based on normalized speaker attribution.
                existing_stated_pids = {e.target_id for e in existing_stated_edges}
                desired_stated_pids = (
                    {desired_stated_by} if desired_stated_by else set()
                )

                for edge in existing_stated_edges:
                    if edge.target_id not in desired_stated_pids:
                        self._graph.remove_edge(edge.id)
                        edges_changed = True

                for pid in desired_stated_pids - existing_stated_pids:
                    self._graph.add_edge(
                        create_stated_by_edge(entry.id, pid, created_by=entry.source)
                    )
                    edges_changed = True

        if count > 0:
            self._persistence.mark_dirty("memories")
            if edges_changed:
                self._persistence.mark_dirty("edges")
            await self._persistence.flush(self._graph)
            if vector_index_changed:
                await self._save_vector_index()

        return count

    async def get_all_memories(self: Store) -> list[MemoryEntry]:
        """Get all memories including archived."""
        return [
            memory.model_copy(deep=True) for memory in self._graph.memories.values()
        ]
