"""Tests for memory doctor missing-provenance prune."""

from ash.cli.commands.memory.doctor.prune_missing_provenance import (
    memory_doctor_prune_missing_provenance,
)
from ash.graph.edges import create_learned_in_edge


async def test_prune_missing_provenance_archives_only_missing(graph_store):
    chat = await graph_store.ensure_chat(
        provider="telegram",
        provider_id="-123",
        chat_type="group",
    )
    missing = await graph_store.add_memory(
        content="Unattributed memory",
        owner_user_id="user-1",
    )
    attributed = await graph_store.add_memory(
        content="Attributed memory",
        chat_id=chat.id,
    )
    graph_store.graph.add_edge(
        create_learned_in_edge(attributed.id, chat.id, created_by="test_setup")
    )
    graph_store._persistence.mark_dirty("edges")
    await graph_store.flush_graph()

    await memory_doctor_prune_missing_provenance(graph_store, force=True)

    missing_after = graph_store.graph.memories[missing.id]
    attributed_after = graph_store.graph.memories[attributed.id]
    assert missing_after.archived_at is not None
    assert missing_after.archive_reason == "missing_provenance"
    assert attributed_after.archived_at is None


async def test_prune_missing_provenance_noop_when_none_missing(graph_store):
    chat = await graph_store.ensure_chat(
        provider="telegram",
        provider_id="-456",
        chat_type="group",
    )
    memory = await graph_store.add_memory(
        content="Already attributed memory",
        chat_id=chat.id,
    )
    graph_store.graph.add_edge(
        create_learned_in_edge(memory.id, chat.id, created_by="test_setup")
    )
    graph_store._persistence.mark_dirty("edges")
    await graph_store.flush_graph()

    await memory_doctor_prune_missing_provenance(graph_store, force=True)

    assert graph_store.graph.memories[memory.id].archived_at is None
