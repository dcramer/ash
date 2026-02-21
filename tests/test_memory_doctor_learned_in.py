"""Tests for memory doctor LEARNED_IN backfill."""

from ash.cli.commands.memory.doctor.backfill_learned_in import (
    memory_doctor_backfill_learned_in,
)
from ash.graph.edges import create_learned_in_edge, get_learned_in_chat


async def test_backfill_learned_in_uses_existing_chat_scope(graph_store):
    chat = await graph_store.ensure_chat(
        provider="telegram",
        provider_id="-12345",
        chat_type="group",
    )
    memory = await graph_store.add_memory(
        content="Team prefers async updates",
        chat_id=chat.id,
    )

    await memory_doctor_backfill_learned_in(graph_store, force=True)

    assert get_learned_in_chat(graph_store.graph, memory.id) == chat.id


async def test_backfill_learned_in_defaults_owner_memories_to_private(graph_store):
    memory = await graph_store.add_memory(
        content="User prefers short replies",
        owner_user_id="user-42",
    )

    await memory_doctor_backfill_learned_in(graph_store, force=True)

    learned_in = get_learned_in_chat(graph_store.graph, memory.id)
    assert learned_in is not None
    chat = graph_store.graph.chats[learned_in]
    assert chat.provider == "legacy-backfill"
    assert chat.provider_id == "owner:user-42"
    assert chat.chat_type == "private"


async def test_backfill_learned_in_skips_already_linked_memories(graph_store):
    chat = await graph_store.ensure_chat(
        provider="telegram",
        provider_id="-99",
        chat_type="group",
    )
    memory = await graph_store.add_memory(
        content="Release cut yesterday", chat_id=chat.id
    )
    graph_store.graph.add_edge(
        create_learned_in_edge(memory.id, chat.id, created_by="test_setup")
    )
    graph_store._persistence.mark_dirty("edges")
    await graph_store.flush_graph()

    before = len(
        [e for e in graph_store.graph.edges.values() if e.edge_type == "LEARNED_IN"]
    )
    await memory_doctor_backfill_learned_in(graph_store, force=True)
    after = len(
        [e for e in graph_store.graph.edges.values() if e.edge_type == "LEARNED_IN"]
    )

    assert before == after
