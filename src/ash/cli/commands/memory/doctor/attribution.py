"""Fix memories missing source_username attribution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ash.cli.commands.memory.doctor._helpers import confirm_or_cancel, truncate
from ash.cli.console import console, create_table, success

if TYPE_CHECKING:
    from ash.store.store import Store


async def memory_doctor_attribution(store: Store, force: bool) -> None:
    """Fix memories missing source_username attribution.

    For personal memories created by agent/cli without source_username,
    infers the speaker from owner_user_id (personal memories = owner spoke).
    """
    memories = await store.list_memories(
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

    table = create_table(
        "Memories to Fix",
        [
            ("ID", {"style": "dim", "max_width": 8}),
            ("Source", "cyan"),
            ("Owner", "green"),
            ("Content", {"style": "white", "max_width": 40}),
        ],
    )

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

    updates = []
    for memory in to_fix:
        updated = memory.model_copy(deep=True)
        updated.source_username = memory.owner_user_id
        updates.append(updated)
    await store.batch_update_memories(updates)

    success(f"Fixed attribution for {len(to_fix)} memories")
