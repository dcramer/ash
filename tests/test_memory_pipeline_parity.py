"""Parity tests between ingestion and doctor repair pipelines."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.cli.commands.memory.doctor.normalize_semantics import (
    memory_doctor_normalize_semantics,
)
from ash.cli.commands.memory.doctor.self_facts import memory_doctor_self_facts
from ash.graph.graph import KnowledgeGraph
from ash.graph.persistence import GraphPersistence
from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.processing import process_extracted_facts
from ash.store.store import Store
from ash.store.types import ExtractedFact, MemoryType


@pytest.fixture
def mock_embedding_generator():
    generator = MagicMock(spec=EmbeddingGenerator)
    generator.embed = AsyncMock(return_value=[0.1] * 1536)
    return generator


@pytest.fixture
def mock_index():
    index = MagicMock()
    index.search = MagicMock(return_value=[])
    index.add = MagicMock()
    index.remove = MagicMock()
    index.save = AsyncMock()
    index.get_ids = MagicMock(return_value=set())
    return index


@pytest.fixture
async def graph_store(graph_dir, mock_index, mock_embedding_generator) -> Store:
    graph = KnowledgeGraph()
    persistence = GraphPersistence(graph_dir)
    store = Store(
        graph=graph,
        persistence=persistence,
        vector_index=mock_index,
        embedding_generator=mock_embedding_generator,
    )
    store._llm_model = "mock-model"
    return store


def _semantic_snapshot(
    store: Store, memory_id: str
) -> tuple[list[str], str | None, str]:
    from ash.graph.edges import get_stated_by_person, get_subject_person_ids
    from ash.store.types import get_assertion

    memory = store.graph.memories[memory_id]
    assertion = get_assertion(memory)
    assert assertion is not None
    return (
        sorted(get_subject_person_ids(store.graph, memory_id)),
        get_stated_by_person(store.graph, memory_id),
        assertion.assertion_kind.value,
    )


class TestMemoryPipelineParity:
    async def test_self_fact_doctor_repair_matches_ingestion(self, graph_store: Store):
        from ash.graph.edges import ABOUT, STATED_BY

        self_person = await graph_store.create_person(
            created_by="user-1",
            name="Alice",
            relationship="self",
            aliases=["alice"],
        )

        ingested_ids = await process_extracted_facts(
            facts=[
                ExtractedFact(
                    content="I prefer tea",
                    subjects=[],
                    shared=False,
                    confidence=0.95,
                    memory_type=MemoryType.PREFERENCE,
                    speaker="alice",
                )
            ],
            store=graph_store,
            user_id="user-1",
            speaker_username="alice",
            speaker_display_name="Alice",
            speaker_person_id=self_person.id,
            owner_names=["alice"],
            source="test",
        )
        assert ingested_ids
        ingested_snapshot = _semantic_snapshot(graph_store, ingested_ids[0])

        repaired = await graph_store.add_memory(
            content="I prefer tea",
            owner_user_id="user-1",
            source_username="alice",
            source="legacy_seed",
        )
        for edge in graph_store.graph.get_outgoing(repaired.id, edge_type=ABOUT):
            graph_store.graph.remove_edge(edge.id)
        for edge in graph_store.graph.get_outgoing(repaired.id, edge_type=STATED_BY):
            graph_store.graph.remove_edge(edge.id)
        graph_store.graph.memories[repaired.id].metadata = None
        graph_store._persistence.mark_dirty("memories", "edges")
        await graph_store.flush_graph()

        await memory_doctor_self_facts(graph_store, force=True)
        await memory_doctor_normalize_semantics(graph_store, force=True)
        repaired_snapshot = _semantic_snapshot(graph_store, repaired.id)

        assert repaired_snapshot == ingested_snapshot

    async def test_person_fact_doctor_repair_matches_ingestion(
        self, graph_store: Store
    ):
        from ash.graph.edges import ABOUT, STATED_BY, create_about_edge

        alice = await graph_store.create_person(
            created_by="user-1",
            name="Alice",
            relationship="self",
            aliases=["alice"],
        )
        bob = await graph_store.create_person(
            created_by="user-1",
            name="Bob",
            aliases=["bob"],
        )

        ingested_ids = await process_extracted_facts(
            facts=[
                ExtractedFact(
                    content="Bob likes coffee",
                    subjects=["Bob"],
                    shared=False,
                    confidence=0.95,
                    memory_type=MemoryType.KNOWLEDGE,
                    speaker="alice",
                )
            ],
            store=graph_store,
            user_id="user-1",
            speaker_username="alice",
            speaker_display_name="Alice",
            speaker_person_id=alice.id,
            owner_names=["alice"],
            source="test",
        )
        assert ingested_ids
        ingested_snapshot = _semantic_snapshot(graph_store, ingested_ids[0])

        repaired = await graph_store.add_memory(
            content="Bob likes coffee",
            owner_user_id="user-1",
            source_username="alice",
            source="legacy_seed",
        )
        for edge in graph_store.graph.get_outgoing(repaired.id, edge_type=ABOUT):
            graph_store.graph.remove_edge(edge.id)
        for edge in graph_store.graph.get_outgoing(repaired.id, edge_type=STATED_BY):
            graph_store.graph.remove_edge(edge.id)
        graph_store.graph.add_edge(
            create_about_edge(repaired.id, bob.id, created_by="legacy_seed")
        )
        graph_store.graph.memories[repaired.id].metadata = None
        graph_store._persistence.mark_dirty("memories", "edges")
        await graph_store.flush_graph()

        await memory_doctor_normalize_semantics(graph_store, force=True)
        repaired_snapshot = _semantic_snapshot(graph_store, repaired.id)

        assert repaired_snapshot == ingested_snapshot
