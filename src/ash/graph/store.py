"""Unified graph store facade.

Replaces MemoryManager + PersonManager with a single store that composes
existing storage layers and adds the GraphIndex for O(1) traversals.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ash.config.paths import (
    get_chats_jsonl_path,
    get_people_jsonl_path,
    get_users_jsonl_path,
)
from ash.graph.index import GraphIndex
from ash.graph.types import ChatEntry, EdgeType, UserEntry
from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.file_store import FileMemoryStore
from ash.memory.index import VectorIndex
from ash.memory.jsonl import TypedJSONL
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
from ash.people.types import (
    AliasEntry,
    PersonEntry,
    PersonResolutionResult,
    RelationshipClaim,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from ash.llm import LLMProvider, LLMRegistry

logger = logging.getLogger(__name__)

CONFLICT_SIMILARITY_THRESHOLD = 0.75

SUPERSESSION_PROMPT = """Given these two memories, determine if the NEW memory supersedes/replaces the OLD memory.

OLD: "{old_content}"
NEW: "{new_content}"

Answer YES if the new memory updates, corrects, or replaces the old memory.
Answer NO if they are about different things or both should be kept.

Answer only YES or NO."""

FUZZY_MATCH_PROMPT = """Given a person reference and a list of known people, determine if the reference matches any existing person.

Reference: "{reference}"
{context_section}
{speaker_section}
Known people:
{people_list}

Consider: name variants (first name <-> full name, nicknames), relationship links (e.g., "Sarah" from speaker "dcramer" matches a person with relationship "wife" stated by "dcramer"), and alias matches. Prefer matching relationships stated by the current speaker.

If the reference clearly refers to one of the known people, respond with ONLY the ID.
If no clear match, respond with NONE.

Respond with only the ID or NONE, nothing else."""

RELATIONSHIP_TERMS = {
    "wife",
    "husband",
    "partner",
    "spouse",
    "mom",
    "mother",
    "dad",
    "father",
    "parent",
    "son",
    "daughter",
    "child",
    "kid",
    "brother",
    "sister",
    "sibling",
    "boss",
    "manager",
    "coworker",
    "colleague",
    "friend",
    "best friend",
    "roommate",
    "doctor",
    "therapist",
    "dentist",
}


class GraphStore:
    """Unified facade replacing MemoryManager + PersonManager.

    Composes existing storage layers (FileMemoryStore, VectorIndex,
    EmbeddingGenerator, TypedJSONL) with a GraphIndex for O(1) traversals.
    """

    def __init__(
        self,
        memory_store: FileMemoryStore,
        vector_index: VectorIndex,
        embedding_generator: EmbeddingGenerator,
        people_path: Path | None = None,
        users_path: Path | None = None,
        chats_path: Path | None = None,
        llm: LLMProvider | None = None,
        max_entries: int | None = None,
    ) -> None:
        self._store = memory_store
        self._index = vector_index
        self._embeddings = embedding_generator
        self._llm = llm
        self._max_entries = max_entries

        # JSONL stores for people, users, chats
        self._people_jsonl = TypedJSONL(
            people_path or get_people_jsonl_path(), PersonEntry
        )
        self._user_jsonl: TypedJSONL[UserEntry] = TypedJSONL(
            users_path or get_users_jsonl_path(), UserEntry
        )
        self._chat_jsonl: TypedJSONL[ChatEntry] = TypedJSONL(
            chats_path or get_chats_jsonl_path(), ChatEntry
        )

        # Graph index
        self._graph = GraphIndex()

        # Caches
        self._people_cache: list[PersonEntry] | None = None
        self._people_mtime: float | None = None
        self._users_cache: list[UserEntry] | None = None
        self._users_mtime: float | None = None
        self._chats_cache: list[ChatEntry] | None = None
        self._chats_mtime: float | None = None
        self._graph_built: bool = False

        # LLM for fuzzy matching (set post-construction)
        self._llm_model: str | None = None

        # Person name cache for memory annotation
        self._person_name_cache: dict[str, str] = {}

        # Memory-by-ID lookup (populated with graph)
        self._memory_by_id: dict[str, MemoryEntry] = {}

    # ------------------------------------------------------------------
    # LLM configuration (set post-construction)
    # ------------------------------------------------------------------

    def set_llm(self, llm: LLMProvider, model: str) -> None:
        """Set LLM provider for fuzzy matching and supersession verification."""
        self._llm = llm
        self._llm_model = model

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    async def _ensure_people_loaded(self) -> list[PersonEntry]:
        current_mtime = (
            self._people_jsonl.get_mtime() if self._people_jsonl.exists() else None
        )
        if self._people_cache is None or current_mtime != self._people_mtime:
            self._people_cache = await self._people_jsonl.load_all()
            self._people_mtime = current_mtime
            self._graph_built = False
        return self._people_cache

    async def _ensure_users_loaded(self) -> list[UserEntry]:
        current_mtime = (
            self._user_jsonl.get_mtime() if self._user_jsonl.exists() else None
        )
        if self._users_cache is None or current_mtime != self._users_mtime:
            self._users_cache = await self._user_jsonl.load_all()
            self._users_mtime = current_mtime
            self._graph_built = False
        return self._users_cache

    async def _ensure_chats_loaded(self) -> list[ChatEntry]:
        current_mtime = (
            self._chat_jsonl.get_mtime() if self._chat_jsonl.exists() else None
        )
        if self._chats_cache is None or current_mtime != self._chats_mtime:
            self._chats_cache = await self._chat_jsonl.load_all()
            self._chats_mtime = current_mtime
            self._graph_built = False
        return self._chats_cache

    async def _ensure_graph_built(self) -> GraphIndex:
        """Ensure graph index is up to date."""
        people = await self._ensure_people_loaded()
        users = await self._ensure_users_loaded()
        chats = await self._ensure_chats_loaded()

        if not self._graph_built:
            memories = await self._store.get_all_memories()
            self._graph.build(memories, people, users, chats)
            self._memory_by_id = {m.id: m for m in memories}
            self._graph_built = True

        return self._graph

    async def _find_memories_about_persons(
        self,
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

        # Collect candidate memory IDs from graph (O(1) per person)
        candidate_ids: set[str] = set()
        for pid in person_ids:
            candidate_ids |= graph.memories_about(pid)

        # Filter candidates using the memory-by-ID dict
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

    def _invalidate_people_cache(self) -> None:
        self._people_cache = None
        self._people_mtime = None
        self._graph_built = False

    def _invalidate_users_cache(self) -> None:
        self._users_cache = None
        self._users_mtime = None
        self._graph_built = False

    def _invalidate_chats_cache(self) -> None:
        self._chats_cache = None
        self._chats_mtime = None
        self._graph_built = False

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------

    async def _resolve_subject_name(self, person_ids: list[str]) -> str | None:
        if not person_ids:
            return None
        for pid in person_ids:
            if pid in self._person_name_cache:
                return self._person_name_cache[pid]
            try:
                person = await self.get_person(pid)
                if person:
                    self._person_name_cache[pid] = person.name
                    return person.name
            except Exception:
                logger.debug("Failed to resolve person name for %s", pid)
        return None

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
        """Add a memory entry."""
        if contains_secret(content):
            raise ValueError(
                "Memory content contains potential secrets and cannot be stored"
            )

        if expires_in_days is not None and expires_at is None:
            from datetime import timedelta

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

        # Persist embedding
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

        self._graph_built = False
        return memory

    async def get_memory(self, memory_id: str) -> MemoryEntry | None:
        await self._ensure_graph_built()
        return self._memory_by_id.get(memory_id)

    async def search(
        self,
        query: str,
        limit: int = 5,
        subject_person_id: str | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[SearchResult]:
        vector_results = await self._index.search(query, limit=limit * 2)
        now = datetime.now(UTC)

        # Ensure memory-by-id cache is populated
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

    async def list_memories(
        self,
        limit: int = 20,
        include_expired: bool = False,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[MemoryEntry]:
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
        memory = await self._store.get_memory_by_prefix(memory_id)
        if not memory:
            return False

        full_id = memory.id
        deleted = await self._store.delete_memory(
            full_id, owner_user_id=owner_user_id, chat_id=chat_id
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

        self._graph_built = False
        return True

    async def gc(self) -> GCResult:
        result = await self._store.gc()
        for memory_id in result.archived_ids:
            try:
                await self._index.delete_embedding(memory_id)
            except Exception:
                logger.debug("Failed to delete embedding for %s", memory_id)
        self._graph_built = False
        return result

    async def enforce_max_entries(self, max_entries: int) -> int:
        now = datetime.now(UTC)
        memories = await self._store.get_memories(
            limit=10000,
            include_expired=False,
            include_superseded=False,
        )

        current_count = len(memories)
        if current_count <= max_entries:
            return 0

        excess = current_count - max_entries
        memories.sort(key=lambda m: m.created_at or datetime.min.replace(tzinfo=UTC))

        evicted = 0
        for memory in memories:
            if evicted >= excess:
                break
            if memory.created_at and (now - memory.created_at).days < 7:
                continue
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
        memories = await self._store.get_all_memories()
        embeddings = await self._store.load_embeddings()
        count = await self._index.rebuild_from_embeddings(memories, embeddings)
        logger.info("index_rebuilt", extra={"count": count})
        return count

    async def remap_subject_person_id(self, old_id: str, new_id: str) -> int:
        count = await self._store.remap_subject_person_id(old_id, new_id)
        if count > 0:
            self._graph_built = False
        return count

    async def forget_person(
        self,
        person_id: str,
        delete_person_record: bool = False,
    ) -> int:
        graph = await self._ensure_graph_built()
        # Use graph index for O(1) lookup of memories about this person
        candidate_ids = graph.memories_about(person_id)
        to_archive = {
            mid
            for mid in candidate_ids
            if (m := self._memory_by_id.get(mid)) and m.archived_at is None
        }

        if not to_archive:
            if delete_person_record:
                await self.delete_person(person_id)
            return 0

        archived_ids = await self._store.archive_memories(to_archive, "forgotten")
        for memory_id in archived_ids:
            try:
                await self._index.delete_embedding(memory_id)
            except Exception:
                logger.debug("Failed to delete embedding for %s", memory_id)

        if delete_person_record:
            await self.delete_person(person_id)

        self._graph_built = False
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
        return await self._store.get_supersession_chain(memory_id)

    async def find_conflicting_memories(
        self,
        new_content: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        subject_person_ids: list[str] | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        similar = await self._index.search(new_content, limit=10)
        now = datetime.now(UTC)

        conflicts: list[tuple[MemoryEntry, float]] = []
        for result in similar:
            if result.similarity < CONFLICT_SIMILARITY_THRESHOLD:
                continue
            memory = await self._store.get_memory(result.memory_id)
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
        self,
        old_content: str,
        new_content: str,
    ) -> bool:
        if not self._llm:
            return True
        try:
            from ash.llm.types import Message, Role

            prompt = SUPERSESSION_PROMPT.format(
                old_content=old_content, new_content=new_content
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
            if await self._mark_superseded(memory.id, new_memory.id):
                count += 1

        return count

    async def _is_protected_by_subject_authority(
        self,
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
            logger.debug("Subject authority check failed", exc_info=True)
            return False

    async def _mark_superseded(self, old_memory_id: str, new_memory_id: str) -> bool:
        success = await self._store.mark_memory_superseded(
            memory_id=old_memory_id, superseded_by_id=new_memory_id
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
            self._graph_built = False
        return success

    async def supersede_confirmed_hearsay(
        self,
        new_memory: MemoryEntry,
        person_ids: set[str],
        source_username: str,
        owner_user_id: str,
        similarity_threshold: float = 0.80,
    ) -> int:
        # Use graph to find candidate memories about these persons
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

        count = 0
        for hearsay in hearsay_candidates:
            if hearsay.id == new_memory.id:
                continue
            try:
                similar = await self._index.search(hearsay.content, limit=5)
                similarity = 0.0
                for result in similar:
                    if result.memory_id == new_memory.id:
                        similarity = result.similarity
                        break
                if similarity < similarity_threshold:
                    continue
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

    def _passes_privacy_filter(
        self,
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

    async def get_context_for_message(
        self,
        user_id: str,
        user_message: str,
        chat_id: str | None = None,
        max_memories: int = 10,
        chat_type: str | None = None,
        participant_person_ids: dict[str, set[str]] | None = None,
    ) -> RetrievedContext:
        try:
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

        # Cross-context retrieval
        if participant_person_ids:
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
                                    similarity=0.7,
                                    metadata=cross_meta,
                                    source_type="memory",
                                )
                            )
                except Exception:
                    logger.debug(
                        "Failed to get cross-context memories for %s", username
                    )

        # Second pass: graph traversal
        seen_ids: set[str] = {m.id for m in memories}
        mentioned_person_ids: set[str] = set()
        for m in memories:
            spids = (m.metadata or {}).get("subject_person_ids") or []
            mentioned_person_ids.update(spids)

        if participant_person_ids:
            for pids in participant_person_ids.values():
                mentioned_person_ids -= pids

        if mentioned_person_ids:
            try:
                subject_cross = await self._find_memories_about_persons(
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
                        cross_meta = {
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

    # ------------------------------------------------------------------
    # Person operations
    # ------------------------------------------------------------------

    async def create_person(
        self,
        created_by: str,
        name: str,
        relationship: str | None = None,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        relationship_stated_by: str | None = None,
    ) -> PersonEntry:
        now = datetime.now(UTC)
        relationships: list[RelationshipClaim] = []
        if relationship:
            relationships.append(
                RelationshipClaim(
                    relationship=relationship,
                    stated_by=relationship_stated_by or created_by,
                    created_at=now,
                )
            )
        alias_entries = [
            AliasEntry(value=a, added_by=created_by, created_at=now)
            for a in (aliases or [])
        ]
        entry = PersonEntry(
            id=str(uuid.uuid4()),
            version=1,
            created_by=created_by,
            name=name,
            relationships=relationships,
            aliases=alias_entries,
            created_at=now,
            updated_at=now,
            metadata=metadata,
        )
        await self._people_jsonl.append(entry)
        self._invalidate_people_cache()
        logger.debug(
            "person_created", extra={"person_id": entry.id, "person_name": name}
        )
        return entry

    async def get_person(self, person_id: str) -> PersonEntry | None:
        people = await self._ensure_people_loaded()
        return self._find_person_by_id(people, person_id)

    async def find_person(self, reference: str) -> PersonEntry | None:
        ref = self._normalize_reference(reference)
        people = await self._ensure_people_loaded()
        for person in people:
            if person.merged_into:
                continue
            if person.name.lower() == ref:
                return person
            for rc in person.relationships:
                if rc.relationship.lower() == ref:
                    return person
            for alias in person.aliases:
                if self._normalize_reference(alias.value) == ref:
                    return person
        return None

    async def find_person_for_speaker(
        self,
        reference: str,
        speaker_user_id: str,
    ) -> PersonEntry | None:
        ref = self._normalize_reference(reference)
        people = await self._ensure_people_loaded()
        for person in people:
            if person.merged_into:
                continue
            is_connected = any(
                rc.stated_by == speaker_user_id for rc in person.relationships
            ) or any(ae.added_by == speaker_user_id for ae in person.aliases)
            if not is_connected:
                continue
            if person.name.lower() == ref:
                return person
            for rc in person.relationships:
                if rc.relationship.lower() == ref:
                    return person
            for alias in person.aliases:
                if self._normalize_reference(alias.value) == ref:
                    return person
        return None

    async def list_people(self) -> list[PersonEntry]:
        people = await self._ensure_people_loaded()
        result = [p for p in people if not p.merged_into]
        result.sort(key=lambda x: x.name)
        return result

    async def get_all_people(self) -> list[PersonEntry]:
        return await self._ensure_people_loaded()

    async def update_person(
        self,
        person_id: str,
        name: str | None = None,
        relationship: str | None = None,
        aliases: list[str] | None = None,
        updated_by: str | None = None,
    ) -> PersonEntry | None:
        people = await self._ensure_people_loaded()
        person = self._find_person_by_id(people, person_id)
        if not person:
            return None
        now = datetime.now(UTC)
        if name is not None:
            person.name = name
        if relationship is not None:
            person.relationships = [
                RelationshipClaim(
                    relationship=relationship, stated_by=updated_by, created_at=now
                )
            ]
        if aliases is not None:
            person.aliases = [
                AliasEntry(value=a, added_by=updated_by, created_at=now)
                for a in aliases
            ]
        person.updated_at = now
        await self._people_jsonl.rewrite(people)
        self._invalidate_people_cache()
        return person

    async def add_alias(
        self,
        person_id: str,
        alias: str,
        added_by: str | None = None,
    ) -> PersonEntry | None:
        people = await self._ensure_people_loaded()
        person = self._find_person_by_id(people, person_id)
        if not person:
            return None
        existing_values = [a.value.lower() for a in person.aliases]
        if alias.lower() not in existing_values:
            person.aliases.append(
                AliasEntry(value=alias, added_by=added_by, created_at=datetime.now(UTC))
            )
            person.updated_at = datetime.now(UTC)
            await self._people_jsonl.rewrite(people)
            self._invalidate_people_cache()
        return person

    async def add_relationship(
        self,
        person_id: str,
        relationship: str,
        stated_by: str | None = None,
    ) -> PersonEntry | None:
        people = await self._ensure_people_loaded()
        person = self._find_person_by_id(people, person_id)
        if not person:
            return None
        existing_rels = [r.relationship.lower() for r in person.relationships]
        if relationship.lower() not in existing_rels:
            person.relationships.append(
                RelationshipClaim(
                    relationship=relationship,
                    stated_by=stated_by,
                    created_at=datetime.now(UTC),
                )
            )
            person.updated_at = datetime.now(UTC)
            await self._people_jsonl.rewrite(people)
            self._invalidate_people_cache()
        return person

    async def delete_person(self, person_id: str) -> bool:
        people = await self._ensure_people_loaded()
        person = self._find_person_by_id(people, person_id)
        if not person:
            return False
        people.remove(person)
        for p in people:
            if p.merged_into == person_id:
                p.merged_into = None
        await self._people_jsonl.rewrite(people)
        self._invalidate_people_cache()
        logger.debug(
            "person_deleted", extra={"person_id": person_id, "person_name": person.name}
        )
        return True

    async def merge_people(
        self,
        primary_id: str,
        secondary_id: str,
    ) -> PersonEntry | None:
        people = await self._ensure_people_loaded()
        primary = self._find_person_by_id(people, primary_id)
        secondary = self._find_person_by_id(people, secondary_id)
        if not primary or not secondary:
            return None
        if secondary.merged_into:
            logger.debug(
                "Skipping merge: secondary %s already merged into %s",
                secondary_id,
                secondary.merged_into,
            )
            return None

        existing_values = {a.value.lower() for a in primary.aliases}
        for alias in secondary.aliases:
            if alias.value.lower() not in existing_values:
                primary.aliases.append(alias)
                existing_values.add(alias.value.lower())
        if (
            secondary.name.lower() != primary.name.lower()
            and secondary.name.lower() not in existing_values
        ):
            primary.aliases.append(
                AliasEntry(
                    value=secondary.name, added_by=None, created_at=datetime.now(UTC)
                )
            )
        existing_rels = {r.relationship.lower() for r in primary.relationships}
        for rc in secondary.relationships:
            if rc.relationship.lower() not in existing_rels:
                primary.relationships.append(rc)
                existing_rels.add(rc.relationship.lower())

        secondary.merged_into = primary_id
        primary.updated_at = datetime.now(UTC)
        await self._people_jsonl.rewrite(people)
        self._invalidate_people_cache()

        logger.debug(
            "person_merged",
            extra={"primary_id": primary_id, "secondary_id": secondary_id},
        )

        # Auto-remap memory references
        try:
            remapped = await self.remap_subject_person_id(secondary_id, primary_id)
            if remapped:
                logger.debug(
                    "Remapped %d memories from %s to %s",
                    remapped,
                    secondary_id,
                    primary_id,
                )
        except Exception:
            logger.debug("Failed to remap memories after merge", exc_info=True)

        return primary

    async def resolve_or_create_person(
        self,
        created_by: str,
        reference: str,
        content_hint: str | None = None,
        relationship_stated_by: str | None = None,
    ) -> PersonResolutionResult:
        # Speaker-scoped search first
        if relationship_stated_by:
            speaker_match = await self.find_person_for_speaker(
                reference, relationship_stated_by
            )
            if speaker_match:
                person = await self._follow_merge_chain(speaker_match)
                return PersonResolutionResult(
                    person_id=person.id, created=False, person_name=person.name
                )

        existing = await self.find_person(reference)
        if existing:
            person = await self._follow_merge_chain(existing)
            return PersonResolutionResult(
                person_id=person.id, created=False, person_name=person.name
            )

        # Try fuzzy match
        fuzzy_match = await self._fuzzy_find(
            reference,
            content_hint=content_hint,
            speaker=relationship_stated_by or created_by,
        )
        if fuzzy_match:
            person = await self._follow_merge_chain(fuzzy_match)
            await self.add_alias(person.id, reference, added_by=created_by)
            logger.debug(
                "fuzzy_match_resolved",
                extra={
                    "reference": reference,
                    "person_id": person.id,
                    "person_name": person.name,
                },
            )
            return PersonResolutionResult(
                person_id=person.id, created=False, person_name=person.name
            )

        name, relationship = self._parse_person_reference(reference, content_hint)
        person = await self.create_person(
            created_by=created_by,
            name=name,
            relationship=relationship,
            aliases=[reference] if reference.lower() != name.lower() else None,
            relationship_stated_by=relationship_stated_by,
        )
        return PersonResolutionResult(
            person_id=person.id, created=True, person_name=person.name
        )

    async def resolve_names(self, person_ids: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for pid in person_ids:
            person = await self.get_person(pid)
            if person:
                result[pid] = person.name
        return result

    async def find_person_ids_for_username(self, username: str) -> set[str]:
        username_clean = username.lstrip("@").lower()

        # Fast path: use graph to resolve username -> user -> person via IS_PERSON
        graph = await self._ensure_graph_built()
        user_node_id = graph.resolve_user_by_username(username_clean)
        if user_node_id:
            person_ids = graph.neighbors(user_node_id, EdgeType.IS_PERSON, "outgoing")
            if person_ids:
                # Follow merge chains
                result: set[str] = set()
                for pid in person_ids:
                    person = await self.get_person(pid)
                    if person and person.merged_into:
                        primary = await self._follow_merge_chain(person)
                        result.add(primary.id)
                    elif person:
                        result.add(person.id)
                return result

        # Fallback: linear scan over people (handles cases where no user node exists)
        people = await self._ensure_people_loaded()
        matching: set[str] = set()
        for person in people:
            if self._matches_username(person, username_clean):
                if person.merged_into:
                    primary = await self._follow_merge_chain(person)
                    matching.add(primary.id)
                else:
                    matching.add(person.id)
        return matching

    async def find_dedup_candidates(
        self,
        person_ids: list[str],
        *,
        exclude_self: bool = False,
    ) -> list[tuple[str, str]]:
        if not self._llm or not self._llm_model:
            return []
        people = await self._ensure_people_loaded()
        active = [p for p in people if not p.merged_into]
        new_people = [p for p in active if p.id in set(person_ids)]
        if not new_people:
            return []

        seen: set[frozenset[str]] = set()
        candidates: list[tuple[PersonEntry, PersonEntry]] = []
        for new_person in new_people:
            for existing in active:
                if existing.id == new_person.id:
                    continue
                pair_key = frozenset({new_person.id, existing.id})
                if pair_key in seen:
                    continue
                if exclude_self:
                    a_rels = {r.relationship.lower() for r in new_person.relationships}
                    b_rels = {r.relationship.lower() for r in existing.relationships}
                    if "self" in a_rels or "self" in b_rels:
                        continue
                if self._heuristic_match(new_person, existing):
                    seen.add(pair_key)
                    candidates.append((new_person, existing))

        if not candidates:
            return []

        results: list[tuple[str, str]] = []
        for person_a, person_b in candidates:
            if await self._llm_verify_same_person(person_a, person_b):
                primary_id, secondary_id = self._pick_primary(person_a, person_b)
                results.append((primary_id, secondary_id))
        return results

    # ------------------------------------------------------------------
    # User operations
    # ------------------------------------------------------------------

    async def ensure_user(
        self,
        provider: str,
        provider_id: str,
        username: str | None = None,
        display_name: str | None = None,
        person_id: str | None = None,
    ) -> UserEntry:
        """Upsert a user node. Creates if not found, updates if changed."""
        users = await self._ensure_users_loaded()
        now = datetime.now(UTC)

        for user in users:
            if user.provider == provider and user.provider_id == provider_id:
                changed = False
                if username is not None and user.username != username:
                    user.username = username
                    changed = True
                if display_name is not None and user.display_name != display_name:
                    user.display_name = display_name
                    changed = True
                if person_id is not None and user.person_id != person_id:
                    user.person_id = person_id
                    changed = True
                if changed:
                    user.updated_at = now
                    await self._user_jsonl.rewrite(users)
                    self._invalidate_users_cache()
                return user

        # Create new user
        entry = UserEntry(
            id=str(uuid.uuid4()),
            version=1,
            provider=provider,
            provider_id=provider_id,
            username=username,
            display_name=display_name,
            person_id=person_id,
            created_at=now,
            updated_at=now,
        )
        await self._user_jsonl.append(entry)
        self._invalidate_users_cache()
        logger.debug(
            "user_created",
            extra={
                "user_id": entry.id,
                "provider": provider,
                "provider_id": provider_id,
            },
        )
        return entry

    async def get_user(self, user_id: str) -> UserEntry | None:
        users = await self._ensure_users_loaded()
        for u in users:
            if u.id == user_id:
                return u
        return None

    async def find_user_by_provider(
        self, provider: str, provider_id: str
    ) -> UserEntry | None:
        users = await self._ensure_users_loaded()
        for u in users:
            if u.provider == provider and u.provider_id == provider_id:
                return u
        return None

    async def list_users(self) -> list[UserEntry]:
        return await self._ensure_users_loaded()

    # ------------------------------------------------------------------
    # Chat operations
    # ------------------------------------------------------------------

    async def ensure_chat(
        self,
        provider: str,
        provider_id: str,
        chat_type: str | None = None,
        title: str | None = None,
    ) -> ChatEntry:
        """Upsert a chat node. Creates if not found, updates if changed."""
        chats = await self._ensure_chats_loaded()
        now = datetime.now(UTC)

        for chat in chats:
            if chat.provider == provider and chat.provider_id == provider_id:
                changed = False
                if chat_type is not None and chat.chat_type != chat_type:
                    chat.chat_type = chat_type
                    changed = True
                if title is not None and chat.title != title:
                    chat.title = title
                    changed = True
                if changed:
                    chat.updated_at = now
                    await self._chat_jsonl.rewrite(chats)
                    self._invalidate_chats_cache()
                return chat

        entry = ChatEntry(
            id=str(uuid.uuid4()),
            version=1,
            provider=provider,
            provider_id=provider_id,
            chat_type=chat_type,
            title=title,
            created_at=now,
            updated_at=now,
        )
        await self._chat_jsonl.append(entry)
        self._invalidate_chats_cache()
        logger.debug(
            "chat_created",
            extra={
                "chat_id": entry.id,
                "provider": provider,
                "provider_id": provider_id,
            },
        )
        return entry

    async def get_chat(self, chat_id: str) -> ChatEntry | None:
        chats = await self._ensure_chats_loaded()
        for c in chats:
            if c.id == chat_id:
                return c
        return None

    async def find_chat_by_provider(
        self, provider: str, provider_id: str
    ) -> ChatEntry | None:
        chats = await self._ensure_chats_loaded()
        for c in chats:
            if c.provider == provider and c.provider_id == provider_id:
                return c
        return None

    async def list_chats(self) -> list[ChatEntry]:
        return await self._ensure_chats_loaded()

    # ------------------------------------------------------------------
    # Graph traversal operations
    # ------------------------------------------------------------------

    async def get_graph(self) -> GraphIndex:
        """Get the graph index, rebuilding if needed."""
        return await self._ensure_graph_built()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_person_by_id(
        people: list[PersonEntry], person_id: str
    ) -> PersonEntry | None:
        for p in people:
            if p.id == person_id:
                return p
        return None

    async def _follow_merge_chain(self, person: PersonEntry) -> PersonEntry:
        visited: set[str] = set()
        current = person
        while current.merged_into and current.merged_into not in visited:
            visited.add(current.id)
            next_person = await self.get_person(current.merged_into)
            if not next_person:
                break
            current = next_person
        return current

    @staticmethod
    def _normalize_reference(text: str) -> str:
        result = text.lower().strip()
        for prefix in ["my ", "the ", "@"]:
            if result.startswith(prefix):
                result = result[len(prefix) :]
        return result

    @staticmethod
    def _matches_username(person: PersonEntry, username: str) -> bool:
        username_lower = username.lower()
        if person.name.lower() == username_lower:
            return True
        return any(alias.value.lower() == username_lower for alias in person.aliases)

    @staticmethod
    def _heuristic_match(a: PersonEntry, b: PersonEntry) -> bool:
        a_rels = {r.relationship.lower() for r in a.relationships}
        b_rels = {r.relationship.lower() for r in b.relationships}
        if "self" in a_rels and "self" in b_rels:
            return False
        a_aliases = {alias.value.lower() for alias in a.aliases}
        b_aliases = {alias.value.lower() for alias in b.aliases}
        if a_aliases & b_aliases:
            return True
        a_name = a.name.lower()
        b_name = b.name.lower()
        if a_name in b_rels or b_name in a_rels:
            return True
        if len(a_name) >= 3 and len(b_name) >= 3:
            if a_name in b_name or b_name in a_name:
                return True
        a_parts = set(a_name.split())
        b_parts = set(b_name.split())
        if (len(a_parts) == 1 or len(b_parts) == 1) and a_parts & b_parts:
            return True
        return False

    async def _llm_verify_same_person(self, a: PersonEntry, b: PersonEntry) -> bool:
        if not self._llm or not self._llm_model:
            return False
        try:
            from ash.llm.types import Message, Role

            a_aliases = ", ".join(al.value for al in a.aliases) or "none"
            b_aliases = ", ".join(al.value for al in b.aliases) or "none"
            a_rels = ", ".join(r.relationship for r in a.relationships) or "none"
            b_rels = ", ".join(r.relationship for r in b.relationships) or "none"
            prompt = (
                "Do these two person records refer to the same real-world person?\n\n"
                f"Person A: Name: {a.name}, Aliases: {a_aliases}, Relationships: {a_rels}\n"
                f"Person B: Name: {b.name}, Aliases: {b_aliases}, Relationships: {b_rels}\n\n"
                "Answer only YES or NO."
            )
            response = await self._llm.complete(
                messages=[Message(role=Role.USER, content=prompt)],
                model=self._llm_model,
                max_tokens=10,
                temperature=0.0,
            )
            return response.message.get_text().strip().upper() == "YES"
        except Exception:
            logger.debug("llm_verify_same_person_failed", exc_info=True)
            return False

    @staticmethod
    def _pick_primary(a: PersonEntry, b: PersonEntry) -> tuple[str, str]:
        _epoch = datetime.min.replace(tzinfo=UTC)

        def _sort_key(p: PersonEntry) -> tuple[int, datetime]:
            score = len(p.aliases) + len(p.relationships)
            return (-score, p.created_at or _epoch)

        first, second = sorted([a, b], key=_sort_key)
        return first.id, second.id

    async def _fuzzy_find(
        self,
        reference: str,
        content_hint: str | None = None,
        speaker: str | None = None,
    ) -> PersonEntry | None:
        if not self._llm or not self._llm_model:
            return None
        people = await self._ensure_people_loaded()
        candidates = [p for p in people if not p.merged_into]
        if not candidates:
            return None
        try:
            from ash.llm.types import Message, Role

            lines = []
            for p in candidates:
                parts = [f"ID: {p.id}, Name: {p.name}"]
                if p.aliases:
                    alias_strs = [a.value for a in p.aliases]
                    parts.append(f"Aliases: {', '.join(alias_strs)}")
                if p.relationships:
                    rel_parts = []
                    for r in p.relationships:
                        if r.stated_by:
                            rel_parts.append(
                                f"{r.relationship} (stated by {r.stated_by})"
                            )
                        else:
                            rel_parts.append(r.relationship)
                    parts.append(f"Relationships: {', '.join(rel_parts)}")
                lines.append(" | ".join(parts))

            people_list = "\n".join(lines)
            context_section = f'Context: "{content_hint}"' if content_hint else ""
            speaker_section = f'Speaker: "{speaker}"' if speaker else ""
            prompt = FUZZY_MATCH_PROMPT.format(
                reference=reference,
                people_list=people_list,
                context_section=context_section,
                speaker_section=speaker_section,
            )
            response = await self._llm.complete(
                messages=[Message(role=Role.USER, content=prompt)],
                model=self._llm_model,
                max_tokens=50,
                temperature=0.0,
            )
            result = response.message.get_text().strip()
            if result == "NONE":
                return None
            return self._find_person_by_id(candidates, result)
        except Exception:
            logger.debug("fuzzy_find_failed", exc_info=True)
            return None

    def _parse_person_reference(
        self,
        reference: str,
        content_hint: str | None = None,
    ) -> tuple[str, str | None]:
        ref_lower = reference.lower().strip()
        if ref_lower.startswith("@"):
            ref_lower = ref_lower[1:]
        relationship = ref_lower[3:] if ref_lower.startswith("my ") else None
        if relationship and relationship in RELATIONSHIP_TERMS:
            if content_hint:
                name = self._extract_name_from_content(content_hint, relationship)
                if name:
                    return name, relationship
            return relationship.title(), relationship
        return ref_lower.title(), None

    @staticmethod
    def _extract_name_from_content(content: str, relationship: str) -> str | None:
        import re

        def _extract_capitalized_name(text: str) -> str | None:
            match = re.match(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
            return match.group(1) if match else None

        match = re.search(
            rf"{relationship}(?:'s name is| is named)\s+",
            content,
            re.IGNORECASE,
        )
        if match:
            name = _extract_capitalized_name(content[match.end() :])
            if name:
                return name

        match = re.search(
            rf"(?:^|,\s*)my {relationship}\s+",
            content,
            re.IGNORECASE,
        )
        if match:
            name = _extract_capitalized_name(content[match.end() :])
            if name:
                return name

        match = re.search(r"^(\w+)'s\s", content)
        if match:
            name = match.group(1)
            if name.lower() not in ["user", "my", "the", "their", "his", "her"]:
                return name
        return None


async def create_graph_store(
    db_session: AsyncSession,
    llm_registry: LLMRegistry,
    embedding_model: str | None = None,
    embedding_provider: str = "openai",
    max_entries: int | None = None,
    llm_provider: str = "anthropic",
    auto_migrate: bool = True,
) -> GraphStore:
    """Create a fully-wired GraphStore.

    Replaces both create_memory_manager() and create_person_manager().
    """
    from ash.graph.migration import migrate_filesystem
    from ash.memory.migration import (
        check_db_has_memories,
        migrate_db_to_jsonl,
        migrate_to_graph_dir,
        needs_migration,
    )

    # Filesystem restructure migration
    if auto_migrate:
        try:
            migrate_filesystem()
        except Exception:
            logger.warning("Filesystem migration failed", exc_info=True)

    # Old scattered paths -> graph/ migration
    if auto_migrate:
        try:
            if await migrate_to_graph_dir():
                logger.info("Migrated to graph directory layout")
        except Exception:
            logger.warning("Graph directory migration failed", exc_info=True)

    # SQLite -> JSONL migration
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

    # Get LLM for supersession verification
    llm = None
    try:
        llm = llm_registry.get(llm_provider)
    except Exception:
        logger.debug("LLM not available for supersession verification")

    # Check if index needs rebuild
    embeddings = await store.load_embeddings()
    indexed_count = await index.get_embedding_count()
    if embeddings and indexed_count == 0:
        logger.info("Index empty, rebuilding from embeddings.jsonl")
        memories = await store.get_all_memories()
        await index.rebuild_from_embeddings(memories, embeddings)

    graph_store = GraphStore(
        memory_store=store,
        vector_index=index,
        embedding_generator=embedding_generator,
        llm=llm,
        max_entries=max_entries,
    )

    # Data migration: extract User/Chat nodes from existing memories
    if auto_migrate:
        try:
            from ash.graph.migration import migrate_data_to_graph

            users_created, chats_created = await migrate_data_to_graph(
                memory_store=store,
                user_jsonl=graph_store._user_jsonl,
                chat_jsonl=graph_store._chat_jsonl,
                people_jsonl=graph_store._people_jsonl,
            )
            if users_created or chats_created:
                logger.info(
                    "Data migration: %d users, %d chats", users_created, chats_created
                )
        except Exception:
            logger.warning("Data migration failed", exc_info=True)

    return graph_store
