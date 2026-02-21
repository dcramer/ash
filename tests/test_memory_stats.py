from datetime import UTC, datetime

from ash.cli.commands.memory.stats import collect_memory_stats
from ash.graph.edges import create_learned_in_edge


async def test_collect_memory_stats_reads_graph_and_state(graph_store):
    chat = await graph_store.ensure_chat(
        provider="telegram",
        provider_id="-300",
        chat_type="group",
    )
    active = await graph_store.add_memory(
        content="Active memory",
        chat_id=chat.id,
    )
    graph_store.graph.add_edge(
        create_learned_in_edge(active.id, chat.id, created_by="test_setup")
    )
    graph_store._persistence.mark_dirty("edges")
    await graph_store.flush_graph()

    archived = await graph_store.add_memory(
        content="Archived memory",
        owner_user_id="user-1",
    )
    archived.archived_at = datetime.now(UTC)
    graph_store._persistence.mark_dirty("memories")
    await graph_store.flush_graph()

    await graph_store._persistence.update_state(
        provenance_missing_count=2,
        vector_missing_count=3,
        vector_removed_extra_count=1,
        consistency_checked_at="2026-02-21T03:12:51+00:00",
    )

    stats = await collect_memory_stats(graph_store)

    assert stats["active_memories"] == 1
    assert stats["total_memories"] == 2
    assert stats["provenance_missing_count"] == 2
    assert stats["vector_missing_count"] == 3
    assert stats["vector_removed_extra_count"] == 1
    assert stats["consistency_checked_at"] == "2026-02-21T03:12:51+00:00"


async def test_collect_memory_stats_defaults_missing_state_values(graph_store):
    stats = await collect_memory_stats(graph_store)

    assert stats["active_memories"] == 0
    assert stats["total_memories"] == 0
    assert stats["provenance_missing_count"] == 0
    assert stats["vector_missing_count"] == 0
    assert stats["vector_removed_extra_count"] == 0
    assert stats["consistency_checked_at"] is None
