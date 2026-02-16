"""Backfill subject_person_ids by matching content against known people."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.table import Table

from ash.cli.commands.memory.doctor._helpers import confirm_or_cancel, truncate
from ash.cli.console import console, success
from ash.core.filters import build_owner_matchers, is_owner_name
from ash.store.types import MemoryType

if TYPE_CHECKING:
    from ash.store.store import Store
    from ash.store.types import MemoryEntry


async def memory_doctor_backfill_subjects(store: Store, force: bool) -> None:
    """Backfill subject_person_ids by matching content against known people.

    Scans memories with empty subject_person_ids for references to known
    person names in their content. When a match is found (and the person
    isn't the speaker), the memory is linked to that person.

    This fixes cases like "David Cramer likes mayo" where the content
    mentions a person by name but subject_person_ids was never populated.
    """
    memories = await store.list_memories(
        limit=None, include_expired=True, include_superseded=True
    )

    from ash.graph.edges import get_subject_person_ids

    # Candidates: no subject links, not RELATIONSHIP type
    candidates = [
        m
        for m in memories
        if not get_subject_person_ids(store.graph, m.id)
        and m.memory_type != MemoryType.RELATIONSHIP
    ]

    if not candidates:
        success("No memories need subject backfill")
        return

    # Build name/alias → person_id lookup from all people
    people = await store.list_people()
    name_to_person: dict[str, str] = {}
    for person in people:
        name_lower = person.name.lower()
        name_to_person[name_lower] = person.id
        for alias in person.aliases:
            name_to_person[alias.value.lower()] = person.id
        # For multi-word names, also index name parts (first/last)
        parts = name_lower.split()
        if len(parts) > 1:
            for part in parts:
                if len(part) >= 3 and part not in name_to_person:
                    name_to_person[part] = person.id

    if not name_to_person:
        success("No people found; nothing to backfill")
        return

    # Sort names by length descending so longer names match first
    sorted_names: list[str] = sorted(
        name_to_person.keys(), key=lambda n: len(n), reverse=True
    )

    # Match candidates to people by scanning content
    to_fix: list[tuple[MemoryEntry, str, str]] = []  # (memory, person_id, matched_name)
    for memory in candidates:
        content_lower = memory.content.lower()

        # Build owner matchers for the speaker to avoid self-linking
        speaker_names: list[str] = []
        if memory.source_username:
            speaker_names.append(memory.source_username)

        owner_matchers = build_owner_matchers(speaker_names) if speaker_names else None

        matched_person_id: str | None = None
        matched_name: str | None = None
        for name in sorted_names:
            if name not in content_lower:
                continue
            person_id = name_to_person[name]

            # Skip if this name refers to the speaker (avoid double-linking self-facts)
            if owner_matchers and is_owner_name(name, owner_matchers):
                continue

            matched_person_id = person_id
            matched_name = name
            break

        if matched_person_id and matched_name:
            to_fix.append((memory, matched_person_id, matched_name))

    if not to_fix:
        success("No memories need subject backfill")
        return

    table = Table(title="Memories Missing Subject Attribution")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Matched", style="green")
    table.add_column("Person ID", style="cyan", max_width=8)
    table.add_column("Content", style="white", max_width=40)

    for memory, person_id, matched_name in to_fix[:10]:
        table.add_row(
            memory.id[:8],
            matched_name,
            person_id[:8],
            truncate(memory.content),
        )

    if len(to_fix) > 10:
        table.add_row("...", "...", "...", f"... and {len(to_fix) - 10} more")

    console.print(table)
    console.print(f"\n[bold]{len(to_fix)} memories need subject backfill[/bold]")

    if not confirm_or_cancel("Backfill subject_person_ids for these memories?", force):
        return

    # Build subject_person_ids_map for batch update
    subject_person_ids_map = {memory.id: [person_id] for memory, person_id, _ in to_fix}
    await store.batch_update_memories(
        [m for m, _, _ in to_fix], subject_person_ids_map=subject_person_ids_map
    )

    success(f"Backfilled subject_person_ids for {len(to_fix)} memories")
