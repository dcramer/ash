"""Fix memories missing source_username attribution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.table import Table

from ash.cli.commands.memory.doctor._helpers import confirm_or_cancel, truncate
from ash.cli.console import console, success

if TYPE_CHECKING:
    from ash.graph.store import GraphStore


async def memory_doctor_attribution(graph_store: GraphStore, force: bool) -> None:
    """Fix memories missing source_username attribution.

    For personal memories created by agent/cli without source_username,
    infers the speaker from owner_user_id (personal memories = owner spoke).
    """
    memories = await graph_store.list_memories(
        limit=None, include_expired=True, include_superseded=True
    )

    to_fix = [
        m
        for m in memories
        if m.source in ("agent", "cli", "rpc")
        and not m.source_username
        and m.owner_user_id
    ]

    if not to_fix:
        success("No memories need attribution fix")
        return

    table = Table(title="Memories to Fix")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Source", style="cyan")
    table.add_column("Owner", style="green")
    table.add_column("Content", style="white", max_width=40)

    for memory in to_fix[:10]:
        table.add_row(
            memory.id[:8],
            memory.source or "-",
            memory.owner_user_id or "-",
            truncate(memory.content),
        )

    if len(to_fix) > 10:
        table.add_row("...", "...", "...", f"... and {len(to_fix) - 10} more")

    console.print(table)
    console.print(f"\n[bold]{len(to_fix)} memories need attribution fix[/bold]")

    if not confirm_or_cancel("Fix attribution for these memories?", force):
        return

    for memory in to_fix:
        memory.source_username = memory.owner_user_id
    await graph_store.batch_update_memories(to_fix)

    success(f"Fixed attribution for {len(to_fix)} memories")
