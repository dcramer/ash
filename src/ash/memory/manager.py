"""Memory manager for orchestrating retrieval and persistence.

This module provides the primary facade for all memory operations,
using filesystem-primary storage with a SQLite vector index.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.file_store import FileMemoryStore
from ash.memory.index import VectorIndex
from ash.memory.secrets import contains_secret
from ash.memory.types import (
    GCResult,
    MemoryEntry,
    MemoryType,
    RetrievedContext,
    SearchResult,
    Sensitivity,
    matches_scope,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from ash.llm import LLMProvider, LLMRegistry
    from ash.people import PersonManager

logger = logging.getLogger(__name__)

CONFLICT_SIMILARITY_THRESHOLD = 0.75

# LLM verification prompt for supersession
SUPERSESSION_PROMPT = """Given these two memories, determine if the NEW memory supersedes/replaces the OLD memory.

OLD: "{old_content}"
NEW: "{new_content}"

Answer YES if the new memory updates, corrects, or replaces the old memory.
Answer NO if they are about different things or both should be kept.

Answer only YES or NO."""


class MemoryManager:
    """Orchestrates memory retrieval and persistence.

    Uses filesystem-primary storage (JSONL files) with a SQLite
    vector index for semantic search. The JSONL is the source of
    truth and the index is rebuildable.
    """

    def __init__(
        self,
        store: FileMemoryStore,
        index: VectorIndex,
        embedding_generator: EmbeddingGenerator,
        llm: LLMProvider | None = None,
        max_entries: int | None = None,
        person_manager: PersonManager | None = None,
    ) -> None:
        """Initialize memory manager.

        Args:
            store: Filesystem-based memory store.
            index: Vector index for semantic search.
            embedding_generator: Generator for embeddings.
            llm: LLM for supersession verification (optional).
            max_entries: Optional cap on active memories.
            person_manager: Optional person manager for subject authority checks.
        """
        self._store = store
        self._index = index
        self._embeddings = embedding_generator
        self._llm = llm
        self._max_entries = max_entries
        self._people = person_manager
        # Cache resolved person names within a single request
        self._person_name_cache: dict[str, str] = {}

    async def _resolve_subject_name(self, person_ids: list[str]) -> str | None:
        """Resolve subject person IDs to a display name for memory annotation.

        Returns the first resolvable name, or None if no person manager or
        no match. Results are cached for the lifetime of the manager instance.
        """
        if not self._people or not person_ids:
            return None
        for pid in person_ids:
            if pid in self._person_name_cache:
                return self._person_name_cache[pid]
            try:
                person = await self._people.get(pid)
                if person:
                    self._person_name_cache[pid] = person.name
                    return person.name
            except Exception:
                logger.debug("Failed to resolve person name for %s", pid)
        return None

    async def get_context_for_message(
        self,
        user_id: str,
        user_message: str,
        chat_id: str | None = None,
        max_memories: int = 10,
        chat_type: str | None = None,
        participant_person_ids: dict[str, set[str]] | None = None,
    ) -> RetrievedContext:
        """Retrieve relevant memory context before LLM call.

        Supports cross-context retrieval: facts learned about a person in public
        chats are recallable in private chats with that person, subject to
        sensitivity filtering.

        Args:
            user_id: User ID for scoping.
            user_message: Message to find relevant context for.
            chat_id: Optional chat ID for group memories.
            max_memories: Maximum memories to return.
            chat_type: Type of chat ("private", "group", "supergroup").
            participant_person_ids: Map of username -> person_ids (pre-resolved).

        Returns:
            Retrieved context with matching memories.
        """
        try:
            # Get user's own memories
            memories = await self.search(
                query=user_message,
                limit=max_memories,
                owner_user_id=user_id,
                chat_id=chat_id,
            )
        except Exception:
            logger.warning(
                "Failed to search memories, continuing without", exc_info=True
            )
            memories = []

        # Cross-context retrieval: get memories about participants from other owners
        if participant_person_ids:
            for username, person_ids in participant_person_ids.items():
                if not person_ids:
                    continue
                try:
                    cross_memories = await self._store.find_memories_by_subject(
                        person_ids=person_ids,
                        exclude_owner_user_id=user_id,  # Don't double-count
                        limit=max_memories,
                    )

                    # Apply privacy filtering and convert to SearchResult
                    for memory in cross_memories:
                        if self._passes_privacy_filter(
                            sensitivity=memory.sensitivity,
                            subject_person_ids=memory.subject_person_ids,
                            chat_type=chat_type,
                            querying_person_ids=person_ids,
                        ):
                            subject_name = await self._resolve_subject_name(
                                memory.subject_person_ids
                            )
                            cross_meta: dict[str, Any] = {
                                "memory_type": memory.memory_type.value,
                                "subject_person_ids": memory.subject_person_ids,
                                "cross_context": True,
                            }
                            if subject_name:
                                cross_meta["subject_name"] = subject_name
                            memories.append(
                                SearchResult(
                                    id=memory.id,
                                    content=memory.content,
                                    similarity=0.7,  # Default for cross-context
                                    metadata=cross_meta,
                                    source_type="memory",
                                )
                            )
                except Exception:
                    logger.debug(
                        "Failed to get cross-context memories for %s", username
                    )

        # Note: privacy filtering is NOT applied to the owner's own memories
        # (from primary search). Those are already scope-checked by search().
        # Cross-context memories have their own filter applied above (lines 142-148).
        # Filtering own memories would incorrectly block SENSITIVE self-memories
        # (no subjects → is_subject always False) and PERSONAL memories about
        # others (subject != owner → filtered out in group chats).

        # --- Second pass: graph traversal ---
        # Collect person IDs mentioned in primary and cross-context results.
        # These represent people "in scope" for the conversation that we should
        # also retrieve facts about (e.g., David's wife Sukhpreet → get Sukhpreet's facts).
        seen_ids: set[str] = {m.id for m in memories}
        mentioned_person_ids: set[str] = set()
        for m in memories:
            spids = (m.metadata or {}).get("subject_person_ids") or []
            mentioned_person_ids.update(spids)

        # Remove the querying user's own person IDs to avoid self-retrieval loops
        if participant_person_ids:
            for pids in participant_person_ids.values():
                mentioned_person_ids -= pids

        if mentioned_person_ids:
            try:
                subject_cross = await self._store.find_memories_by_subject(
                    person_ids=mentioned_person_ids,
                    exclude_owner_user_id=user_id,
                    limit=max_memories,
                )
                for memory in subject_cross:
                    if memory.id in seen_ids:
                        continue
                    querying_person_ids: set[str] = set()
                    if participant_person_ids:
                        for pids in participant_person_ids.values():
                            querying_person_ids |= pids
                    if self._passes_privacy_filter(
                        sensitivity=memory.sensitivity,
                        subject_person_ids=memory.subject_person_ids,
                        chat_type=chat_type,
                        querying_person_ids=querying_person_ids,
                    ):
                        subject_name = await self._resolve_subject_name(
                            memory.subject_person_ids
                        )
                        cross_meta: dict[str, Any] = {
                            "memory_type": memory.memory_type.value,
                            "subject_person_ids": memory.subject_person_ids,
                            "cross_context": True,
                            "graph_traversal": True,
                        }
                        if subject_name:
                            cross_meta["subject_name"] = subject_name
                        memories.append(
                            SearchResult(
                                id=memory.id,
                                content=memory.content,
                                similarity=0.6,
                                metadata=cross_meta,
                                source_type="memory",
                            )
                        )
                        seen_ids.add(memory.id)
            except Exception:
                logger.debug("Graph traversal cross-context failed", exc_info=True)

        # Deduplicate and limit
        unique_memories: list[SearchResult] = []
        final_seen: set[str] = set()
        for m in memories:
            if m.id not in final_seen:
                final_seen.add(m.id)
                unique_memories.append(m)
                if len(unique_memories) >= max_memories:
                    break

        return RetrievedContext(memories=unique_memories)

    def _passes_privacy_filter(
        self,
        sensitivity: Sensitivity | None,
        subject_person_ids: list[str],
        chat_type: str | None,
        querying_person_ids: set[str],
    ) -> bool:
        """Filter memories based on sensitivity and context.

        Args:
            sensitivity: Privacy classification of the memory.
            subject_person_ids: Person IDs the memory is about.
            chat_type: Type of chat ("private", "group", etc.).
            querying_person_ids: Pre-resolved person IDs for the querying user.

        Returns:
            True if memory passes the privacy filter.
        """
        if sensitivity is None or sensitivity == Sensitivity.PUBLIC:
            return True

        is_subject = bool(set(subject_person_ids) & querying_person_ids)

        if sensitivity == Sensitivity.PERSONAL:
            return is_subject

        if sensitivity == Sensitivity.SENSITIVE:
            return chat_type == "private" and is_subject

        return False

    async def add_memory(
        self,
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
    ) -> MemoryEntry:
        """Add a memory entry.

        Args:
            content: Memory content.
            source: Origin tracking (user, extraction, cli, rpc).
            memory_type: Type classification (defaults to KNOWLEDGE).
            expires_at: Explicit expiration time.
            expires_in_days: Days until expiration (converted to expires_at).
            owner_user_id: Personal scope owner.
            chat_id: Group scope.
            subject_person_ids: People this memory is about.
            observed_at: When fact was observed (for extraction).
            source_username: Who said/provided this fact (handle/username).
            source_display_name: Display name of source user.
            source_session_id: Session ID for extraction tracing.
            source_message_id: Message ID for extraction tracing.
            extraction_confidence: Confidence score for extraction.
            sensitivity: Privacy classification (default PUBLIC).
            metadata: Additional metadata.

        Returns:
            The created memory entry.

        Raises:
            ValueError: If content contains potential secrets.
        """
        # Reject secrets before storing (defense in depth)
        if contains_secret(content):
            raise ValueError(
                "Memory content contains potential secrets and cannot be stored"
            )

        if expires_in_days is not None and expires_at is None:
            expires_at = datetime.now(UTC) + timedelta(days=expires_in_days)

        if memory_type is None:
            memory_type = MemoryType.KNOWLEDGE

        # Generate embedding
        embedding_floats: list[float] | None = None
        try:
            embedding_floats = await self._embeddings.embed(content)
        except Exception:
            logger.warning(
                "Failed to generate embedding, continuing without", exc_info=True
            )

        # Create memory entry
        memory = await self._store.add_memory(
            content=content,
            memory_type=memory_type,
            source=source,
            expires_at=expires_at,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            subject_person_ids=subject_person_ids,
            observed_at=observed_at,
            source_username=source_username,
            source_display_name=source_display_name,
            source_session_id=source_session_id,
            source_message_id=source_message_id,
            extraction_confidence=extraction_confidence,
            sensitivity=sensitivity,
            portable=portable,
            metadata=metadata,
        )

        # Persist embedding to JSONL and index
        if embedding_floats:
            embedding_base64 = MemoryEntry.encode_embedding(embedding_floats)
            try:
                await self._store.save_embedding(memory.id, embedding_base64)
            except Exception:
                logger.warning(
                    "Failed to save embedding to JSONL, continuing", exc_info=True
                )
            try:
                await self._index.add_embedding(memory.id, embedding_floats)
            except Exception:
                logger.warning("Failed to index memory, continuing", exc_info=True)

        # Check for conflicting memories
        try:
            superseded_count = await self.supersede_conflicting_memories(
                new_memory=memory,
                owner_user_id=owner_user_id,
                chat_id=chat_id,
            )
            if superseded_count > 0:
                logger.debug(
                    "Memory superseded %d older entries",
                    superseded_count,
                    extra={"new_memory_id": memory.id},
                )
        except Exception:
            logger.warning("Failed to check for conflicting memories", exc_info=True)

        # Enforce max entries
        if self._max_entries is not None:
            try:
                evicted = await self.enforce_max_entries(self._max_entries)
                if evicted > 0:
                    logger.info(
                        "Evicted memories to enforce limit",
                        extra={"evicted": evicted, "max_entries": self._max_entries},
                    )
            except Exception:
                logger.warning("Failed to enforce max_entries limit", exc_info=True)

        return memory

    async def get_memory(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory by ID.

        Args:
            memory_id: Memory UUID.

        Returns:
            Memory entry or None.
        """
        return await self._store.get_memory(memory_id)

    async def find_conflicting_memories(
        self,
        new_content: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        subject_person_ids: list[str] | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Find existing memories that may conflict with new content.

        Args:
            new_content: Content of new memory.
            owner_user_id: User scope.
            chat_id: Chat scope.
            subject_person_ids: Subject filter.

        Returns:
            List of (memory, similarity) tuples above threshold.
        """
        similar = await self._index.search(new_content, limit=10)
        now = datetime.now(UTC)

        conflicts: list[tuple[MemoryEntry, float]] = []
        for result in similar:
            if result.similarity < CONFLICT_SIMILARITY_THRESHOLD:
                continue

            memory = await self._store.get_memory(result.memory_id)
            if not memory:
                continue

            # Skip expired or superseded
            if memory.superseded_at:
                continue
            if memory.expires_at and memory.expires_at <= now:
                continue

            # Scope check
            if not matches_scope(memory, owner_user_id, chat_id):
                continue

            # Subject matching
            memory_subjects = memory.subject_person_ids or []
            if subject_person_ids:
                if not set(subject_person_ids) & set(memory_subjects):
                    continue
            elif memory_subjects:
                continue

            conflicts.append((memory, result.similarity))

        return conflicts

    async def _verify_conflict_with_llm(
        self,
        old_content: str,
        new_content: str,
    ) -> bool:
        """Verify if new memory truly supersedes old using LLM.

        Args:
            old_content: Content of old memory.
            new_content: Content of new memory.

        Returns:
            True if LLM confirms supersession.
        """
        if not self._llm:
            # Without LLM, trust vector similarity
            return True

        try:
            from ash.llm.types import Message, Role

            prompt = SUPERSESSION_PROMPT.format(
                old_content=old_content,
                new_content=new_content,
            )

            response = await self._llm.complete(
                messages=[Message(role=Role.USER, content=prompt)],
                max_tokens=10,
                temperature=0.0,
            )

            answer = response.message.get_text().strip().upper()
            return answer.startswith("YES")
        except Exception:
            logger.warning(
                "LLM verification failed, falling back to similarity", exc_info=True
            )
            return True

    async def supersede_conflicting_memories(
        self,
        new_memory: MemoryEntry,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> int:
        """Find and mark conflicting memories as superseded.

        Uses vector similarity to find candidates, then optionally
        verifies with LLM before superseding.

        Args:
            new_memory: The new memory that may supersede others.
            owner_user_id: User scope.
            chat_id: Chat scope.

        Returns:
            Number of memories superseded.
        """
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

            # Subject authority: protect self-confirmed facts from third-party supersession
            if await self._is_protected_by_subject_authority(memory, new_memory):
                continue

            # LLM verification for borderline cases
            if similarity < 0.85 and self._llm:
                if not await self._verify_conflict_with_llm(
                    old_content=memory.content,
                    new_content=new_memory.content,
                ):
                    continue

            if await self._mark_superseded(memory.id, new_memory.id):
                count += 1

        return count

    async def _is_protected_by_subject_authority(
        self,
        candidate: MemoryEntry,
        new_memory: MemoryEntry,
    ) -> bool:
        """Check if a candidate memory is protected by subject authority.

        A self-confirmed fact (where source_username resolves to one of the
        subject_person_ids) cannot be superseded by a third party unless the
        third party is also a subject of the memory.

        Only applies when both memories have overlapping subject_person_ids.
        """
        if not self._people:
            return False
        if not candidate.source_username or not candidate.subject_person_ids:
            return False
        if not new_memory.source_username:
            return False
        if new_memory.source_username == candidate.source_username:
            return False

        try:
            # Is the candidate a self-confirmed fact?
            source_ids = await self._people.find_person_ids_for_username(
                candidate.source_username
            )
            if not (source_ids & set(candidate.subject_person_ids)):
                return False

            # Self-confirmed — only allow supersession if new source is also a subject
            new_source_ids = await self._people.find_person_ids_for_username(
                new_memory.source_username
            )
            if new_source_ids & set(candidate.subject_person_ids):
                return False

            return True
        except Exception:
            logger.debug("Subject authority check failed", exc_info=True)
            return False

    async def _mark_superseded(
        self,
        old_memory_id: str,
        new_memory_id: str,
    ) -> bool:
        """Mark a memory as superseded and clean up its embedding.

        Args:
            old_memory_id: Memory to supersede.
            new_memory_id: Memory that supersedes it.

        Returns:
            True if successfully superseded.
        """
        success = await self._store.mark_memory_superseded(
            memory_id=old_memory_id,
            superseded_by_id=new_memory_id,
        )
        if success:
            try:
                await self._index.delete_embedding(old_memory_id)
            except Exception:
                logger.warning(
                    "Failed to delete superseded memory embedding",
                    extra={"memory_id": old_memory_id},
                    exc_info=True,
                )
        return success

    async def search(
        self,
        query: str,
        limit: int = 5,
        subject_person_id: str | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[SearchResult]:
        """Search memories by semantic similarity.

        Args:
            query: Search query.
            limit: Maximum results.
            subject_person_id: Filter to memories about person.
            owner_user_id: User scope.
            chat_id: Chat scope.

        Returns:
            List of search results with similarity scores.
        """
        vector_results = await self._index.search(query, limit=limit * 2)
        now = datetime.now(UTC)

        results: list[SearchResult] = []
        for vr in vector_results:
            memory = await self._store.get_memory(vr.memory_id)
            if not memory:
                continue

            # Skip expired or superseded
            if memory.superseded_at:
                continue
            if memory.expires_at and memory.expires_at <= now:
                continue

            # Scope check
            if not matches_scope(memory, owner_user_id, chat_id):
                continue

            # Subject filter
            if subject_person_id:
                if subject_person_id not in (memory.subject_person_ids or []):
                    continue

            # Resolve subject name for prompt annotation
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

    async def list_memories(
        self,
        limit: int = 20,
        include_expired: bool = False,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[MemoryEntry]:
        """List recent memories without semantic search.

        Args:
            limit: Maximum memories to return.
            include_expired: Include expired entries.
            owner_user_id: User scope.
            chat_id: Chat scope.

        Returns:
            List of memory entries.
        """
        return await self._store.get_memories(
            include_expired=include_expired,
            include_superseded=False,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            limit=limit,
        )

    async def delete_memory(
        self,
        memory_id: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> bool:
        """Delete a memory and its embedding.

        Supports partial memory IDs (prefix matching).

        Args:
            memory_id: Memory UUID or prefix.
            owner_user_id: User for authorization.
            chat_id: Chat for authorization.

        Returns:
            True if memory was deleted.
        """
        memory = await self._store.get_memory_by_prefix(memory_id)
        if not memory:
            return False

        full_id = memory.id

        deleted = await self._store.delete_memory(
            full_id,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
        )
        if not deleted:
            return False

        try:
            await self._index.delete_embedding(full_id)
        except Exception:
            logger.warning(
                "Failed to delete memory embedding",
                extra={"memory_id": full_id},
                exc_info=True,
            )

        return True

    async def gc(self) -> GCResult:
        """Garbage collect expired and superseded memories.

        Uses smart ephemeral decay based on memory type.

        Returns:
            GC result with counts.
        """
        result = await self._store.gc()

        # Delete embeddings for archived memories
        for memory_id in result.archived_ids:
            try:
                await self._index.delete_embedding(memory_id)
            except Exception:
                logger.debug("Failed to delete embedding for %s", memory_id)

        return result

    async def enforce_max_entries(self, max_entries: int) -> int:
        """Evict oldest memories if over the max_entries limit.

        Args:
            max_entries: Maximum allowed active memories.

        Returns:
            Number of memories evicted.
        """
        now = datetime.now(UTC)
        memories = await self._store.get_memories(
            limit=10000,  # Get all
            include_expired=False,
            include_superseded=False,
        )

        current_count = len(memories)
        if current_count <= max_entries:
            return 0

        excess = current_count - max_entries

        # Sort by age (oldest first) for eviction
        memories.sort(key=lambda m: m.created_at or datetime.min.replace(tzinfo=UTC))

        # Evict oldest that are at least 7 days old
        evicted = 0
        for memory in memories:
            if evicted >= excess:
                break

            if memory.created_at and (now - memory.created_at).days < 7:
                continue  # Don't evict recent memories

            await self._store.delete_memory(memory.id)
            try:
                await self._index.delete_embedding(memory.id)
            except Exception:
                logger.debug(
                    "Failed to delete embedding for %s during eviction", memory.id
                )
            evicted += 1

        if evicted < excess:
            logger.warning(
                "Could not evict enough memories - all remaining are recent",
                extra={"excess": excess, "evicted": evicted},
            )

        return evicted

    async def rebuild_index(self) -> int:
        """Rebuild vector index from embeddings.jsonl.

        Returns:
            Number of embeddings indexed.
        """
        memories = await self._store.get_all_memories()
        embeddings = await self._store.load_embeddings()
        count = await self._index.rebuild_from_embeddings(memories, embeddings)
        logger.info("index_rebuilt", extra={"count": count})
        return count

    async def remap_subject_person_id(self, old_id: str, new_id: str) -> int:
        """Replace old_id with new_id in all subject_person_ids.

        Called by the orchestrator after merging person records to keep
        memories reachable via the primary person ID.

        Args:
            old_id: The person ID being merged away.
            new_id: The primary person ID to replace it with.

        Returns:
            Number of memories updated.
        """
        return await self._store.remap_subject_person_id(old_id, new_id)

    async def forget_person(
        self,
        person_id: str,
        delete_person_record: bool = False,
    ) -> int:
        """Archive all memories about a person (the "forget me" operation).

        Finds all memories with ABOUT edges to this person, archives them,
        and removes them from active store and vector index.

        Args:
            person_id: The person ID to forget.
            delete_person_record: If True, also delete the person record.

        Returns:
            Number of memories archived.
        """
        memories = await self._store.get_all_memories()

        to_archive = {
            m.id
            for m in memories
            if person_id in (m.subject_person_ids or []) and m.archived_at is None
        }

        if not to_archive:
            if delete_person_record and self._people:
                await self._people.delete(person_id)
            return 0

        archived_ids = await self._store.archive_memories(to_archive, "forgotten")

        # Remove from vector index
        for memory_id in archived_ids:
            try:
                await self._index.delete_embedding(memory_id)
            except Exception:
                logger.debug("Failed to delete embedding for %s", memory_id)

        # Optionally delete the person record
        if delete_person_record and self._people:
            await self._people.delete(person_id)

        logger.info(
            "forget_person_complete",
            extra={
                "person_id": person_id,
                "archived_count": len(archived_ids),
                "deleted_person": delete_person_record,
            },
        )

        return len(archived_ids)

    async def get_supersession_chain(self, memory_id: str) -> list[MemoryEntry]:
        """Get the chain of superseded memories leading to this ID.

        Args:
            memory_id: Memory to trace back from.

        Returns:
            List of memories in supersession order (oldest first).
        """
        return await self._store.get_supersession_chain(memory_id)

    async def supersede_confirmed_hearsay(
        self,
        new_memory: MemoryEntry,
        person_ids: set[str],
        source_username: str,
        owner_user_id: str,
        similarity_threshold: float = 0.80,
    ) -> int:
        """Supersede hearsay memories when user confirms a fact about themselves.

        When a user states a fact about themselves that was previously reported
        as hearsay by someone else, this method finds and supersedes the hearsay
        memories with the new first-person fact.

        Args:
            new_memory: The new FACT memory (user speaking about themselves).
            person_ids: Pre-resolved person IDs for the source user.
            source_username: Username of the source user.
            owner_user_id: Owner user ID for scoping.
            similarity_threshold: Minimum similarity score to supersede (default 0.80).

        Returns:
            Number of hearsay memories superseded.
        """
        hearsay_candidates = await self._store.find_hearsay_by_subject(
            person_ids=person_ids,
            source_username=source_username,
            owner_user_id=owner_user_id,
        )

        if not hearsay_candidates:
            return 0

        count = 0
        for hearsay in hearsay_candidates:
            if hearsay.id == new_memory.id:
                continue

            # Check similarity using vector search
            try:
                similar = await self._index.search(hearsay.content, limit=5)
                similarity = 0.0
                for result in similar:
                    if result.memory_id == new_memory.id:
                        similarity = result.similarity
                        break

                if similarity < similarity_threshold:
                    continue

                # Use LLM verification for borderline cases
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
                logger.debug(
                    "Failed to check hearsay similarity",
                    extra={"hearsay_id": hearsay.id},
                    exc_info=True,
                )

        return count


async def create_memory_manager(
    db_session: AsyncSession,
    llm_registry: LLMRegistry,
    embedding_model: str | None = None,
    embedding_provider: str = "openai",
    max_entries: int | None = None,
    llm_provider: str = "anthropic",
    auto_migrate: bool = True,
    person_manager: PersonManager | None = None,
) -> MemoryManager:
    """Create a fully-wired MemoryManager.

    Automatically detects and migrates from SQLite to JSONL if needed.

    Args:
        db_session: Database session for vector operations.
        llm_registry: LLM provider registry.
        embedding_model: Embedding model to use.
        embedding_provider: Provider for embeddings (default: openai).
        max_entries: Optional cap on active memories.
        llm_provider: Provider for supersession verification.
        auto_migrate: Whether to auto-migrate from SQLite.
        person_manager: Optional person manager for subject authority checks.

    Returns:
        Configured MemoryManager instance.
    """
    from ash.memory.migration import (
        check_db_has_memories,
        migrate_db_to_jsonl,
        migrate_to_graph_dir,
        needs_migration,
    )

    # Migrate from old scattered paths to graph/ layout
    if auto_migrate:
        try:
            if await migrate_to_graph_dir():
                logger.info("Migrated to graph directory layout")
        except Exception:
            logger.warning("Graph directory migration failed", exc_info=True)

    # Check if SQLite→JSONL migration is needed
    if auto_migrate and needs_migration():
        try:
            has_memories = await check_db_has_memories(db_session)
            if has_memories:
                logger.info("Starting migration from SQLite to JSONL")
                mem_count, people_count = await migrate_db_to_jsonl(db_session)
                logger.info(
                    "Migration complete",
                    extra={"memories": mem_count, "people": people_count},
                )
        except Exception:
            logger.warning("Migration failed, starting fresh", exc_info=True)

    # Create components
    embedding_generator = EmbeddingGenerator(
        registry=llm_registry,
        model=embedding_model,
        provider=embedding_provider,
    )

    store = FileMemoryStore()
    index = VectorIndex(db_session, embedding_generator)
    await index.initialize()

    # Get LLM for supersession verification (optional)
    llm = None
    try:
        llm = llm_registry.get(llm_provider)
    except Exception:
        logger.debug("LLM not available for supersession verification")

    # Check if index needs rebuild from embeddings.jsonl
    embeddings = await store.load_embeddings()
    indexed_count = await index.get_embedding_count()

    if embeddings and indexed_count == 0:
        logger.info("Index empty, rebuilding from embeddings.jsonl")
        memories = await store.get_all_memories()
        await index.rebuild_from_embeddings(memories, embeddings)

    return MemoryManager(
        store=store,
        index=index,
        embedding_generator=embedding_generator,
        llm=llm,
        max_entries=max_entries,
        person_manager=person_manager,
    )
