"""Archive active memories that still lack LEARNED_IN provenance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ash.cli.commands.memory.doctor._helpers import confirm_or_cancel, truncate
from ash.cli.console import console, create_table, success

if TYPE_CHECKING:
    from ash.store.store import Store


async def memory_doctor_prune_missing_provenance(store: Store, force: bool) -> None:
    """Archive active memories with no LEARNED_IN edge."""
    missing_ids = store.missing_learned_in_provenance_ids()
    if not missing_ids:
        success("No active memories are missing LEARNED_IN provenance")
        return

    table = create_table(
        "Active Memories Missing Provenance",
        [
            ("ID", {"style": "dim", "max_width": 8}),
            ("Owner", {"style": "yellow", "max_width": 16}),
            ("Chat", {"style": "cyan", "max_width": 8}),
            ("Content", {"style": "white", "max_width": 42}),
        ],
    )
    for memory_id in missing_ids[:10]:
        memory = store.graph.memories.get(memory_id)
        if memory is None:
            continue
        table.add_row(
            memory.id[:8],
            memory.owner_user_id or "-",
            (memory.chat_id[:8] if memory.chat_id else "-"),
            truncate(memory.content),
        )
    if len(missing_ids) > 10:
        table.add_row("...", "...", "...", f"... and {len(missing_ids) - 10} more")

    console.print(table)
    console.print(
        f"\n[bold]{len(missing_ids)} active memories are missing LEARNED_IN provenance[/bold]"
    )

    if not confirm_or_cancel("Archive these memories?", force):
        return

    archived = await store.archive_memories(
        set(missing_ids), reason="missing_provenance"
    )
    success(f"Archived {len(archived)} memories missing provenance")
