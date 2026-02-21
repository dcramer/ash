"""Memory health and consistency stats command."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ash.cli.console import console, create_table

if TYPE_CHECKING:
    from ash.store.store import Store


async def collect_memory_stats(store: Store) -> dict[str, Any]:
    """Collect memory health stats for operator visibility."""
    active_entries = await store.list_memories(
        limit=None,
        include_expired=False,
        include_superseded=False,
    )
    state = await store._persistence.load_state()
    return {
        "active_memories": len(active_entries),
        "total_memories": len(store.graph.memories),
        "provenance_missing_count": state.get("provenance_missing_count", 0),
        "vector_missing_count": state.get("vector_missing_count", 0),
        "vector_removed_extra_count": state.get("vector_removed_extra_count", 0),
        "consistency_checked_at": state.get("consistency_checked_at"),
    }


async def memory_stats(store: Store) -> None:
    """Show memory health and consistency stats."""
    stats = await collect_memory_stats(store)

    table = create_table(
        "Memory Stats",
        [
            ("Metric", {"style": "cyan", "max_width": 36}),
            ("Value", {"style": "white", "max_width": 48}),
        ],
    )
    table.add_row("Active memories", str(stats["active_memories"]))
    table.add_row("Total memories", str(stats["total_memories"]))
    table.add_row(
        "Missing LEARNED_IN provenance",
        str(stats["provenance_missing_count"]),
    )
    table.add_row("Vector missing active IDs", str(stats["vector_missing_count"]))
    table.add_row("Vector removed extra IDs", str(stats["vector_removed_extra_count"]))
    table.add_row(
        "Last consistency check",
        str(stats["consistency_checked_at"] or "-"),
    )
    console.print(table)
