"""Resolve numeric source_username to display names via people records."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.table import Table

from ash.cli.commands.memory.doctor._helpers import confirm_or_cancel, truncate
from ash.cli.console import console, success

if TYPE_CHECKING:
    from ash.store.store import Store
    from ash.store.types import MemoryEntry, PersonEntry


async def memory_doctor_fix_names(store: Store, force: bool) -> None:
    """Resolve numeric source_username to display names via people records."""
    memories = await store.list_memories(
        limit=None, include_expired=True, include_superseded=True
    )

    to_fix = [
        m
        for m in memories
        if m.source_username
        and m.source_username.isdigit()
        and not m.source_display_name
    ]

    if not to_fix:
        success("No numeric source usernames to resolve")
        return

    all_people = await store.get_all_people()

    # Build mapping: numeric_id -> person (self-relationship from created_by, then aliases)
    numeric_to_person: dict[str, PersonEntry] = {}
    for person in all_people:
        if person.created_by and person.created_by.isdigit():
            for rc in person.relationships:
                if rc.relationship == "self":
                    numeric_to_person[person.created_by] = person
                    break

    for person in all_people:
        for alias in person.aliases:
            if alias.value.isdigit() and alias.value not in numeric_to_person:
                numeric_to_person[alias.value] = person

    # Match memories to people
    fixes: list[tuple[MemoryEntry, PersonEntry]] = []
    for memory in to_fix:
        person = numeric_to_person.get(memory.source_username)  # type: ignore[arg-type]
        if person:
            fixes.append((memory, person))

    if not fixes:
        console.print(
            f"Found {len(to_fix)} memories with numeric usernames but "
            "no matching people records"
        )
        return

    table = Table(title="Numeric Usernames to Resolve")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Numeric ID", style="yellow")
    table.add_column("Display Name", style="green")
    table.add_column("Content", style="white", max_width=40)

    for memory, person in fixes[:15]:
        table.add_row(
            memory.id[:8],
            memory.source_username,
            person.name,
            truncate(memory.content, 50),
        )

    if len(fixes) > 15:
        table.add_row("...", "...", "...", f"... and {len(fixes) - 15} more")

    console.print(table)
    console.print(f"\n[bold]{len(fixes)} memories to update[/bold]")

    if not confirm_or_cancel("Apply name resolution?", force):
        return

    to_update: list[MemoryEntry] = []
    for memory, person in fixes:
        memory.source_display_name = person.name
        # If person has a non-numeric alias, prefer it as the username
        non_numeric_alias = next(
            (alias.value for alias in person.aliases if not alias.value.isdigit()),
            None,
        )
        if non_numeric_alias:
            memory.source_username = non_numeric_alias
        to_update.append(memory)
    if to_update:
        await store.batch_update_memories(to_update)

    success(f"Resolved {len(fixes)} numeric usernames to display names")
