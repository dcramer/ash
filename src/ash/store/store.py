"""Unified store facade backed by in-memory KnowledgeGraph.

All memory, people, user, and chat data in one in-memory graph.
Persistence to JSONL files. Vector search uses numpy.

Implementation is split across focused mixin modules:
- memories: Memory CRUD, eviction, lifecycle
- search: Search, context retrieval, privacy filtering
- supersession: Conflict detection, supersession, hearsay
- people: Person CRUD, resolution, merge, dedup
- users: User/chat CRUD
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ash.graph.graph import KnowledgeGraph
from ash.graph.persistence import GraphPersistence
from ash.graph.vectors import NumpyVectorIndex
from ash.memory.embeddings import EmbeddingGenerator
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
    """Unified facade backed by in-memory KnowledgeGraph.

    All data is stored in-memory dicts. Every mutation updates in-memory
    state and persists to JSONL files atomically. Vector search uses numpy.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        persistence: GraphPersistence,
        vector_index: NumpyVectorIndex,
        embedding_generator: EmbeddingGenerator,
        llm: LLMProvider | None = None,
        max_entries: int | None = None,
    ) -> None:
        self._graph = graph
        self._persistence = persistence
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

    @property
    def graph(self) -> KnowledgeGraph:
        """Public access to the in-memory knowledge graph."""
        return self._graph

    def set_llm(self, llm: LLMProvider, model: str) -> None:
        """Set LLM provider for fuzzy matching and supersession verification."""
        self._llm = llm
        self._llm_model = model

    @property
    def _vector_index_path(self) -> Path:
        return self._persistence.graph_dir / "embeddings" / "memories.npy"

    async def _save_vector_index(self) -> None:
        """Save the vector index to disk."""
        try:
            await self._index.save(self._vector_index_path)
            await self._persistence.update_state(
                vector_commit_id=f"v-{uuid.uuid4().hex}",
                vector_id_hash=self._vector_id_hash(),
            )
        except Exception:
            logger.warning("vector_index_save_failed", exc_info=True)

    async def _remove_from_vector_index(self, memory_ids: list[str]) -> None:
        """Remove memory IDs from vector index and save."""
        for mid in memory_ids:
            try:
                self._index.remove(mid)
            except Exception:
                logger.warning("embedding_removal_failed", extra={"memory.id": mid})
        if memory_ids:
            await self._save_vector_index()

    def _vector_id_hash(self) -> str:
        """Stable hash of vector index IDs."""
        ids = sorted(self._index.get_ids())
        payload = "\n".join(ids).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _active_memory_id_hash(self) -> str:
        """Stable hash of active memory IDs in the graph."""
        now = datetime.now(UTC)
        active = sorted(
            mid for mid, memory in self._graph.memories.items() if memory.is_active(now)
        )
        payload = "\n".join(active).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    async def ensure_vector_consistency(
        self,
        *,
        auto_rebuild_missing: bool = False,
    ) -> tuple[int, int]:
        """Repair index drift and report (removed_extra, missing_after_repair)."""
        now = datetime.now(UTC)
        active_ids = {
            mid for mid, memory in self._graph.memories.items() if memory.is_active(now)
        }
        index_ids = self._index.get_ids()

        extra_ids = sorted(index_ids - active_ids)
        if extra_ids:
            for mid in extra_ids:
                try:
                    self._index.remove(mid)
                except Exception:
                    logger.warning(
                        "vector_index_extra_remove_failed",
                        extra={"memory.id": mid},
                        exc_info=True,
                    )
            await self._save_vector_index()

        missing_ids = sorted(active_ids - self._index.get_ids())
        if missing_ids:
            logger.warning(
                "vector_index_missing_active_memories",
                extra={"count": len(missing_ids), "memory.ids": missing_ids[:20]},
            )
            if auto_rebuild_missing:
                generated = await self.rebuild_index()
                logger.info(
                    "vector_index_rebuild_startup",
                    extra={"generated_count": generated},
                )
                missing_ids = sorted(active_ids - self._index.get_ids())

        await self._persistence.update_state(
            active_memory_id_hash=self._active_memory_id_hash(),
            vector_id_hash=self._vector_id_hash(),
            vector_missing_count=len(missing_ids),
        )
        return len(extra_ids), len(missing_ids)


async def create_store(
    graph_dir: Path,
    llm_registry: LLMRegistry,
    embedding_model: str | None = None,
    embedding_provider: str = "openai",
    max_entries: int | None = None,
    llm_provider: str = "openai",
) -> Store:
    """Create a fully-wired Store.

    Loads KnowledgeGraph from JSONL, NumpyVectorIndex from .npy files.
    """
    from ash.graph.backfill import backfill_edges_from_raw
    from ash.graph.persistence import hydrate_graph

    # Load raw JSONL and hydrate into typed graph
    persistence = GraphPersistence(graph_dir)
    raw_data = await persistence.load_raw()
    graph = hydrate_graph(raw_data)

    # Backfill edges from legacy FK fields if edges.jsonl is empty
    if not graph.edges and (graph.memories or graph.people or graph.users):
        result = backfill_edges_from_raw(
            graph,
            raw_data["raw_memories"],
            raw_data["raw_people"],
            raw_data["raw_users"],
        )
        if result.created > 0:
            logger.info("edges_backfilled", extra={"count": result.created})
            if result.skipped:
                for msg in result.skipped:
                    logger.warning("backfill_warning", extra={"error.message": msg})
            persistence.mark_dirty("edges")
            await persistence.flush(graph)

    # Load vector index
    embeddings_dir = graph_dir / "embeddings"
    npy_path = embeddings_dir / "memories.npy"
    vector_index = await NumpyVectorIndex.load(npy_path)

    embedding_generator = EmbeddingGenerator(
        registry=llm_registry,
        model=embedding_model,
        provider=embedding_provider,
    )

    llm = None
    try:
        llm = llm_registry.get(llm_provider)
    except Exception:
        logger.debug("LLM not available for supersession verification")

    store = Store(
        graph=graph,
        persistence=persistence,
        vector_index=vector_index,
        embedding_generator=embedding_generator,
        llm=llm,
        max_entries=max_entries,
    )

    # Repair index drift caused by partial writes or external edits.
    auto_rebuild_missing = os.getenv("ASH_MEMORY_AUTO_REBUILD_INDEX", "").lower() in (
        "1",
        "true",
        "yes",
    )
    removed, missing = await store.ensure_vector_consistency(
        auto_rebuild_missing=auto_rebuild_missing
    )
    if removed or missing:
        logger.warning(
            "vector_index_consistency_check",
            extra={
                "removed_extra": removed,
                "missing_after_repair": missing,
                "auto_rebuild_missing": auto_rebuild_missing,
            },
        )
    else:
        logger.debug("vector_index_consistency_ok")

    return store
