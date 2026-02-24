"""Tests for assertion-backed memory semantics."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.cli.commands.memory.doctor.normalize_semantics import (
    memory_doctor_normalize_semantics,
)
from ash.graph.graph import KnowledgeGraph
from ash.graph.persistence import GraphPersistence
from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.processing import process_extracted_facts
from ash.store.store import Store
from ash.store.types import (
    AssertionEnvelope,
    AssertionKind,
    AssertionPredicate,
    ExtractedFact,
    MemoryType,
    PredicateObjectType,
    get_assertion,
)


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


class TestAssertionPipeline:
    async def test_process_extracted_facts_writes_self_assertion(
        self, graph_store: Store
    ):
        from ash.graph.edges import get_stated_by_person, get_subject_person_ids

        person = await graph_store.create_person(
            created_by="user-1",
            name="Alice",
            relationship="self",
            aliases=["alice"],
        )

        fact = ExtractedFact(
            content="I prefer tea",
            subjects=[],
            shared=False,
            confidence=0.95,
            memory_type=MemoryType.PREFERENCE,
            speaker="alice",
        )

        stored_ids = await process_extracted_facts(
            facts=[fact],
            store=graph_store,
            user_id="user-1",
            speaker_username="alice",
            speaker_display_name="Alice",
            speaker_person_id=person.id,
            owner_names=["alice"],
            source="test",
        )

        assert stored_ids
        memory = await graph_store.get_memory(stored_ids[0])
        assert memory is not None

        assertion = get_assertion(memory)
        assert assertion is not None
        assert assertion.assertion_kind == AssertionKind.SELF_FACT
        assert assertion.subjects == [person.id]
        assert get_subject_person_ids(graph_store.graph, memory.id) == [person.id]
        assert get_stated_by_person(graph_store.graph, memory.id) == person.id

    async def test_add_memory_uses_assertion_subjects_for_edges(
        self, graph_store: Store
    ):
        from ash.graph.edges import get_subject_person_ids

        p1 = await graph_store.create_person(created_by="user-1", name="Alice")
        p2 = await graph_store.create_person(created_by="user-1", name="Bob")

        assertion = AssertionEnvelope(
            assertion_kind=AssertionKind.PERSON_FACT,
            subjects=[p2.id],
            predicates=[
                AssertionPredicate(
                    name="describes",
                    object_type=PredicateObjectType.TEXT,
                    value="Bob likes cycling",
                )
            ],
        )

        memory = await graph_store.add_memory(
            content="Bob likes cycling",
            owner_user_id="user-1",
            subject_person_ids=[p1.id],
            assertion=assertion,
        )

        assert get_subject_person_ids(graph_store.graph, memory.id) == [p2.id]
        stored_assertion = get_assertion(memory)
        assert stored_assertion is not None
        assert stored_assertion.subjects == [p2.id]

    async def test_search_returns_assertion_summary_metadata(
        self, graph_store: Store, mock_index
    ):
        person = await graph_store.create_person(created_by="user-1", name="Bob")

        memory = await graph_store.add_memory(
            content="Bob likes espresso",
            owner_user_id="user-1",
            assertion=AssertionEnvelope(
                assertion_kind=AssertionKind.PERSON_FACT,
                subjects=[person.id],
                speaker_person_id=person.id,
                predicates=[
                    AssertionPredicate(
                        name="describes",
                        object_type=PredicateObjectType.TEXT,
                        value="Bob likes espresso",
                    )
                ],
            ),
        )

        mock_index.search.return_value = [(memory.id, 0.91)]

        results = await graph_store.search(
            query="espresso",
            limit=5,
            owner_user_id="user-1",
        )

        assert results
        metadata = results[0].metadata or {}
        assert metadata.get("assertion_kind") == "person_fact"
        assert metadata.get("speaker_person_id") == person.id

    async def test_normalize_semantics_adopts_edge_subjects_for_existing_assertion(
        self, graph_store: Store
    ):
        from ash.graph.edges import create_about_edge, get_subject_person_ids

        person = await graph_store.create_person(created_by="user-1", name="Sarah")

        memory = await graph_store.add_memory(
            content="Sarah's birthday is August 12",
            owner_user_id="user-1",
            assertion=AssertionEnvelope(
                assertion_kind=AssertionKind.CONTEXT_FACT,
                subjects=[],
                predicates=[
                    AssertionPredicate(
                        name="describes",
                        object_type=PredicateObjectType.TEXT,
                        value="Sarah's birthday is August 12",
                    )
                ],
            ),
        )

        graph_store.graph.add_edge(create_about_edge(memory.id, person.id))
        graph_store._persistence.mark_dirty("edges")
        await graph_store.flush_graph()

        await memory_doctor_normalize_semantics(graph_store, force=True)

        updated = graph_store.graph.memories[memory.id]
        assertion = get_assertion(updated)
        assert assertion is not None
        assert assertion.subjects == [person.id]
        assert assertion.assertion_kind == AssertionKind.PERSON_FACT
        assert get_subject_person_ids(graph_store.graph, memory.id) == [person.id]
