"""Unified graph store facade.

Replaces MemoryManager + PersonManager with a single store that composes
existing storage layers and adds the GraphIndex for O(1) traversals.

Implementation is split across focused mixin modules:
- memory_ops: Memory CRUD, eviction, lifecycle
- search: Search, context retrieval, privacy filtering
- supersession: Conflict detection, supersession, hearsay
- people_ops: Person CRUD, resolution, merge, dedup
- user_chat_ops: User/chat CRUD, graph traversal
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ash.config.paths import (
    get_chats_jsonl_path,
    get_people_jsonl_path,
    get_users_jsonl_path,
)
from ash.graph.index import GraphIndex
from ash.graph.memory_ops import MemoryOpsMixin
from ash.graph.people_ops import RELATIONSHIP_TERMS as RELATIONSHIP_TERMS
from ash.graph.people_ops import PeopleOpsMixin
from ash.graph.search import SearchMixin
from ash.graph.supersession import SupersessionMixin
from ash.graph.types import ChatEntry, UserEntry
from ash.graph.user_chat_ops import UserChatOpsMixin
from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.file_store import FileMemoryStore
from ash.memory.index import VectorIndex
from ash.memory.jsonl import TypedJSONL
from ash.memory.types import MemoryEntry
from ash.people.types import PersonEntry

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from ash.llm import LLMProvider, LLMRegistry

logger = logging.getLogger(__name__)


class GraphStore(
    MemoryOpsMixin,
    SearchMixin,
    SupersessionMixin,
    PeopleOpsMixin,
    UserChatOpsMixin,
):
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

        self._people_jsonl = TypedJSONL(
            people_path or get_people_jsonl_path(), PersonEntry
        )
        self._user_jsonl: TypedJSONL[UserEntry] = TypedJSONL(
            users_path or get_users_jsonl_path(), UserEntry
        )
        self._chat_jsonl: TypedJSONL[ChatEntry] = TypedJSONL(
            chats_path or get_chats_jsonl_path(), ChatEntry
        )

        self._graph = GraphIndex()
        self._graph_built: bool = False

        # Write locks for JSONL read-modify-write cycles
        self._people_write_lock = asyncio.Lock()
        self._user_write_lock = asyncio.Lock()
        self._chat_write_lock = asyncio.Lock()
        self._self_person_lock = asyncio.Lock()

        # Caches
        self._people_cache: list[PersonEntry] | None = None
        self._people_mtime: float | None = None
        self._users_cache: list[UserEntry] | None = None
        self._users_mtime: float | None = None
        self._chats_cache: list[ChatEntry] | None = None
        self._chats_mtime: float | None = None
        self._person_name_cache: dict[str, str] = {}
        self._memory_by_id: dict[str, MemoryEntry] = {}

        # LLM model for fuzzy matching (set via set_llm post-construction)
        self._llm_model: str | None = None

    def set_llm(self, llm: LLMProvider, model: str) -> None:
        """Set LLM provider for fuzzy matching and supersession verification."""
        self._llm = llm
        self._llm_model = model

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

    if auto_migrate:
        try:
            migrate_filesystem()
        except Exception:
            logger.warning("Filesystem migration failed", exc_info=True)

        try:
            if await migrate_to_graph_dir():
                logger.info("Migrated to graph directory layout")
        except Exception:
            logger.warning("Graph directory migration failed", exc_info=True)

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

    embedding_generator = EmbeddingGenerator(
        registry=llm_registry,
        model=embedding_model,
        provider=embedding_provider,
    )

    store = FileMemoryStore()
    index = VectorIndex(db_session, embedding_generator)
    await index.initialize()

    llm = None
    try:
        llm = llm_registry.get(llm_provider)
    except Exception:
        logger.debug("LLM not available for supersession verification")

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
