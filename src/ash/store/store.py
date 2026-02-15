"""Unified store facade backed by SQLite.

All memory, people, user, and chat data in one SQLite database.
Vector search uses sqlite-vec in the same database.

Implementation is split across focused mixin modules:
- memories: Memory CRUD, eviction, lifecycle
- search: Search, context retrieval, privacy filtering
- supersession: Conflict detection, supersession, hearsay
- people: Person CRUD, resolution, merge, dedup
- users: User/chat CRUD
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from ash.db.engine import Database
from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.index import VectorIndex
from ash.store.memories import MemoryOpsMixin
from ash.store.people import RELATIONSHIP_TERMS as RELATIONSHIP_TERMS
from ash.store.people import PeopleOpsMixin
from ash.store.search import SearchMixin
from ash.store.supersession import SupersessionMixin
from ash.store.users import UserChatOpsMixin

if TYPE_CHECKING:
    from ash.llm import LLMProvider, LLMRegistry

logger = logging.getLogger(__name__)


class Store(
    MemoryOpsMixin,
    SearchMixin,
    SupersessionMixin,
    PeopleOpsMixin,
    UserChatOpsMixin,
):
    """Unified facade replacing MemoryManager + PersonManager.

    All data is stored in SQLite tables.  Every mutation is a single-row
    SQL operation.  Edge queries use indexes.  SQLite WAL mode handles
    concurrency.
    """

    def __init__(
        self,
        db: Database,
        vector_index: VectorIndex,
        embedding_generator: EmbeddingGenerator,
        llm: LLMProvider | None = None,
        max_entries: int | None = None,
    ) -> None:
        self._db = db
        self._index = vector_index
        self._embeddings = embedding_generator
        self._llm = llm
        self._max_entries = max_entries

        # LLM model for fuzzy matching (set via set_llm post-construction)
        self._llm_model: str | None = None

        # Optional read cache for person names (avoids repeated lookups)
        self._person_name_cache: dict[str, str] = {}

        # Lock to prevent duplicate self-person creation races
        self._self_person_lock = asyncio.Lock()

    def set_llm(self, llm: LLMProvider, model: str) -> None:
        """Set LLM provider for fuzzy matching and supersession verification."""
        self._llm = llm
        self._llm_model = model


async def create_store(
    db: Database,
    llm_registry: LLMRegistry,
    embedding_model: str | None = None,
    embedding_provider: str = "openai",
    max_entries: int | None = None,
    llm_provider: str = "anthropic",
    auto_migrate: bool = True,
) -> Store:
    """Create a fully-wired Store.

    Replaces both create_memory_manager() and create_person_manager().
    """
    if auto_migrate:
        # Run old filesystem migrations (move files around)
        try:
            from ash.store.migration import migrate_filesystem

            migrate_filesystem()
        except Exception:
            logger.warning("Filesystem migration failed", exc_info=True)

        try:
            from ash.memory.migration import migrate_to_graph_dir

            if await migrate_to_graph_dir():
                logger.info("Migrated to graph directory layout")
        except Exception:
            logger.warning("Graph directory migration failed", exc_info=True)

    embedding_generator = EmbeddingGenerator(
        registry=llm_registry,
        model=embedding_model,
        provider=embedding_provider,
    )

    index = VectorIndex(db, embedding_generator)
    await index.initialize()

    llm = None
    try:
        llm = llm_registry.get(llm_provider)
    except Exception:
        logger.debug("LLM not available for supersession verification")

    if auto_migrate:
        # Migrate JSONL data to SQLite if needed
        try:
            from ash.store.migration_sqlite import migrate_jsonl_to_sqlite

            migrated = await migrate_jsonl_to_sqlite(db)
            if migrated:
                logger.info("Migrated JSONL data to SQLite")
        except Exception:
            logger.warning("JSONL to SQLite migration failed", exc_info=True)

    graph_store = Store(
        db=db,
        vector_index=index,
        embedding_generator=embedding_generator,
        llm=llm,
        max_entries=max_entries,
    )

    return graph_store
