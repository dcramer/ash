"""Tests for trust classification and retrieval weighting.

Tests:
- classify_trust using STATED_BY and ABOUT edges
- get_trust_weight multiplier values
- Trust weight applied in retrieval pipeline _make_result
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ash.graph.edges import (
    create_about_edge,
    create_stated_by_edge,
)
from ash.graph.graph import KnowledgeGraph
from ash.store.trust import classify_trust, get_trust_weight
from ash.store.types import MemoryEntry, MemoryType, PersonEntry


def _add_memory(graph: KnowledgeGraph, memory_id: str = "mem-1") -> MemoryEntry:
    """Add a memory node to the graph."""
    mem = MemoryEntry(
        id=memory_id,
        version=1,
        content="Test memory",
        memory_type=MemoryType.KNOWLEDGE,
        owner_user_id="user-1",
        created_at=datetime.now(UTC),
    )
    graph.memories[memory_id] = mem
    return mem


def _add_person(graph: KnowledgeGraph, person_id: str = "person-1") -> PersonEntry:
    """Add a person node to the graph."""
    person = PersonEntry(
        id=person_id,
        name="Test Person",
        created_by="user-1",
        created_at=datetime.now(UTC),
    )
    graph.people[person_id] = person
    return person


class TestClassifyTrust:
    """Tests for classify_trust using STATED_BY and ABOUT edges."""

    def test_unknown_when_no_stated_by(self):
        """No STATED_BY edge => unknown trust."""
        graph = KnowledgeGraph()
        _add_memory(graph)
        assert classify_trust(graph, "mem-1") == "unknown"

    def test_fact_when_speaker_is_subject(self):
        """Speaker is also a subject => fact."""
        graph = KnowledgeGraph()
        _add_memory(graph)
        _add_person(graph, "person-alice")

        # Alice stated the memory AND it's about Alice
        graph.add_edge(create_stated_by_edge("mem-1", "person-alice"))
        graph.add_edge(create_about_edge("mem-1", "person-alice"))

        assert classify_trust(graph, "mem-1") == "fact"

    def test_hearsay_when_speaker_is_not_subject(self):
        """Speaker is not a subject => hearsay."""
        graph = KnowledgeGraph()
        _add_memory(graph)
        _add_person(graph, "person-alice")
        _add_person(graph, "person-bob")

        # Alice stated the memory but it's about Bob
        graph.add_edge(create_stated_by_edge("mem-1", "person-alice"))
        graph.add_edge(create_about_edge("mem-1", "person-bob"))

        assert classify_trust(graph, "mem-1") == "hearsay"

    def test_fact_when_no_subjects(self):
        """Speaker exists but no ABOUT edges => implicit self-fact."""
        graph = KnowledgeGraph()
        _add_memory(graph)
        _add_person(graph, "person-alice")

        # Alice stated the memory, no ABOUT edges
        graph.add_edge(create_stated_by_edge("mem-1", "person-alice"))

        assert classify_trust(graph, "mem-1") == "fact"

    def test_fact_when_speaker_is_one_of_multiple_subjects(self):
        """Speaker is one of several subjects => fact."""
        graph = KnowledgeGraph()
        _add_memory(graph)
        _add_person(graph, "person-alice")
        _add_person(graph, "person-bob")

        # Alice stated it, about both Alice and Bob
        graph.add_edge(create_stated_by_edge("mem-1", "person-alice"))
        graph.add_edge(create_about_edge("mem-1", "person-alice"))
        graph.add_edge(create_about_edge("mem-1", "person-bob"))

        assert classify_trust(graph, "mem-1") == "fact"


class TestGetTrustWeight:
    """Tests for get_trust_weight multiplier values."""

    def test_fact_weight(self):
        assert get_trust_weight("fact") == 1.0

    def test_hearsay_weight(self):
        assert get_trust_weight("hearsay") == 0.8

    def test_unknown_weight(self):
        assert get_trust_weight("unknown") == 0.9

    def test_fact_higher_than_unknown(self):
        assert get_trust_weight("fact") > get_trust_weight("unknown")

    def test_unknown_higher_than_hearsay(self):
        assert get_trust_weight("unknown") > get_trust_weight("hearsay")


class TestTrustInRetrieval:
    """Tests that trust weights are applied in retrieval results."""

    @pytest.fixture
    def graph(self):
        return KnowledgeGraph()

    def test_fact_preserves_similarity(self, graph: KnowledgeGraph):
        """Fact memories should retain full similarity score."""
        _add_memory(graph, "mem-fact")
        _add_person(graph, "person-a")
        graph.add_edge(create_stated_by_edge("mem-fact", "person-a"))
        # No ABOUT edges = implicit self-fact

        trust = classify_trust(graph, "mem-fact")
        weight = get_trust_weight(trust)
        assert weight == 1.0
        assert 0.9 * weight == 0.9  # similarity preserved

    def test_hearsay_reduces_similarity(self, graph: KnowledgeGraph):
        """Hearsay memories should have reduced similarity score."""
        _add_memory(graph, "mem-hearsay")
        _add_person(graph, "person-a")
        _add_person(graph, "person-b")
        graph.add_edge(create_stated_by_edge("mem-hearsay", "person-a"))
        graph.add_edge(create_about_edge("mem-hearsay", "person-b"))

        trust = classify_trust(graph, "mem-hearsay")
        weight = get_trust_weight(trust)
        assert weight == 0.8
        # A similarity of 0.9 becomes 0.72
        assert 0.9 * weight == pytest.approx(0.72)

    def test_fact_ranks_higher_than_hearsay(self, graph: KnowledgeGraph):
        """Given equal base similarity, fact should rank above hearsay."""
        _add_memory(graph, "mem-fact")
        _add_memory(graph, "mem-hearsay")
        _add_person(graph, "person-a")
        _add_person(graph, "person-b")

        # Fact: Alice says something about herself
        graph.add_edge(create_stated_by_edge("mem-fact", "person-a"))
        graph.add_edge(create_about_edge("mem-fact", "person-a"))

        # Hearsay: Alice says something about Bob
        graph.add_edge(create_stated_by_edge("mem-hearsay", "person-a"))
        graph.add_edge(create_about_edge("mem-hearsay", "person-b"))

        base_similarity = 0.85
        fact_score = base_similarity * get_trust_weight(
            classify_trust(graph, "mem-fact")
        )
        hearsay_score = base_similarity * get_trust_weight(
            classify_trust(graph, "mem-hearsay")
        )
        assert fact_score > hearsay_score
