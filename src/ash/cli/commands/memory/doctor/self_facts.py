"""Fix self-facts missing subject_person_ids attribution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.table import Table

from ash.cli.commands.memory.doctor._helpers import confirm_or_cancel, truncate
from ash.cli.console import console, success
from ash.store.types import MemoryType

if TYPE_CHECKING:
    from ash.store.store import Store
    from ash.store.types import MemoryEntry


async def memory_doctor_self_facts(store: Store, force: bool) -> None:
    """Fix self-facts that are missing subject_person_ids.

    Memories with a source_username but no subject_person_ids (and not
    RELATIONSHIP type) should reference the speaker's self-person record.
    This backfills that link by matching source_username against people
    with a "self" relationship.
    """
    memories = await store.list_memories(
        limit=None, include_expired=True, include_superseded=True
    )

    from ash.graph.edges import get_subject_person_ids

    # Candidates: have source_username, no subject links, not RELATIONSHIP
    candidates = [
        m
        for m in memories
        if m.source_username
        and not get_subject_person_ids(store.graph, m.id)
        and m.memory_type != MemoryType.RELATIONSHIP
    ]

    if not candidates:
        success("No self-facts need subject attribution fix")
        return

    # Build username -> person_id map from self-persons
    people = await store.list_people()
    username_to_person: dict[str, str] = {}
    for person in people:
        is_self = any(r.relationship == "self" for r in person.relationships)
        if not is_self:
            continue
        # Match on name and aliases (lowercased)
        username_to_person[person.name.lower()] = person.id
        for alias in person.aliases:
            username_to_person[alias.value.lower()] = person.id

    if not username_to_person:
        success("No self-persons found; nothing to fix")
        return

    # Match candidates to self-persons
    to_fix: list[tuple[MemoryEntry, str]] = []
    for memory in candidates:
        assert memory.source_username is not None
        person_id = username_to_person.get(memory.source_username.lower())
        if person_id:
            to_fix.append((memory, person_id))

    if not to_fix:
        success("No self-facts need subject attribution fix")
        return

    table = Table(title="Self-Facts Missing Subject Attribution")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Speaker", style="cyan")
    table.add_column("Person ID", style="green", max_width=8)
    table.add_column("Content", style="white", max_width=40)

    for memory, person_id in to_fix[:10]:
        table.add_row(
            memory.id[:8],
            memory.source_username or "-",
            person_id[:8],
            truncate(memory.content),
        )

    if len(to_fix) > 10:
        table.add_row("...", "...", "...", f"... and {len(to_fix) - 10} more")

    console.print(table)
    console.print(
        f"\n[bold]{len(to_fix)} self-facts need subject attribution fix[/bold]"
    )

    if not confirm_or_cancel("Fix subject attribution for these memories?", force):
        return

    # Build subject_person_ids_map for batch update
    subject_person_ids_map = {memory.id: [person_id] for memory, person_id in to_fix}
    await store.batch_update_memories(
        [m for m, _ in to_fix], subject_person_ids_map=subject_person_ids_map
    )

    success(f"Fixed subject attribution for {len(to_fix)} self-facts")
