"""Show and history commands for memory entries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from ash.cli.commands.memory._helpers import is_source_self_reference
from ash.cli.console import console, dim, error

if TYPE_CHECKING:
    from ash.store.store import Store


async def memory_show(store: Store, memory_id: str) -> None:
    """Show full details of a memory entry."""
    from rich.panel import Panel
    from rich.table import Table

    # Find the memory by prefix
    memory = await store.get_memory_by_prefix(memory_id)
    if not memory:
        error(f"No memory found with ID: {memory_id}")
        raise typer.Exit(1)

    # Load people for name lookup
    people = await store.list_people()
    people_by_id = {p.id: p for p in people}

    # Build details table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("ID", memory.id)
    table.add_row("Type", memory.memory_type.value)

    # Source user attribution
    if memory.source_username and memory.source_display_name:
        table.add_row(
            "Source User", f"@{memory.source_username} ({memory.source_display_name})"
        )
    elif memory.source_username:
        table.add_row("Source User", f"@{memory.source_username}")
    else:
        table.add_row("Source User", "-")

    table.add_row("Source Method", memory.source or "-")

    # Scope
    if memory.owner_user_id:
        table.add_row("Scope", f"Personal ({memory.owner_user_id})")
    elif memory.chat_id:
        table.add_row("Scope", f"Group ({memory.chat_id})")
    else:
        table.add_row("Scope", "Global")

    from ash.graph.edges import get_subject_person_ids

    # Subjects (who this memory is about)
    subject_names = []
    subject_person_ids = get_subject_person_ids(store._graph, memory.id)
    if subject_person_ids:
        for person_id in subject_person_ids:
            person = people_by_id.get(person_id)
            if person:
                subject_names.append(f"{person.name} ({person_id[:8]})")
            else:
                subject_names.append(person_id)
        table.add_row("About", ", ".join(subject_names))
    elif memory.source_display_name:
        table.add_row("About", f"{memory.source_display_name} (self)")
    elif memory.source_username:
        table.add_row("About", f"@{memory.source_username} (self)")
    else:
        table.add_row("About", "-")

    # Trust level
    is_self_ref = is_source_self_reference(
        memory.source_username,
        memory.owner_user_id,
        subject_person_ids,
        people,
        people_by_id,
    )
    if not subject_names or is_self_ref:
        table.add_row("Trust", "fact (source speaking about themselves)")
    else:
        table.add_row("Trust", "hearsay (source speaking about others)")

    # Timestamps
    if memory.created_at:
        table.add_row("Created", memory.created_at.isoformat())
    if memory.observed_at and memory.observed_at != memory.created_at:
        table.add_row("Observed", memory.observed_at.isoformat())
    if memory.expires_at:
        table.add_row("Expires", memory.expires_at.isoformat())
    from ash.graph.edges import get_superseded_by

    if memory.superseded_at:
        table.add_row("Superseded", memory.superseded_at.isoformat())
    superseded_by_id = get_superseded_by(store._graph, memory.id)
    if superseded_by_id:
        table.add_row("Superseded By", superseded_by_id)

    # Source attribution
    if memory.source_session_id:
        table.add_row("Session", memory.source_session_id)
    if memory.source_message_id:
        table.add_row("Message", memory.source_message_id)
    if memory.extraction_confidence is not None:
        table.add_row("Confidence", f"{memory.extraction_confidence:.2f}")

    console.print(Panel(table, title=f"Memory {memory.id[:8]}"))
    console.print()
    console.print(Panel(memory.content, title="Content"))


async def memory_history(store: Store, memory_id: str) -> None:
    """Show supersession chain for a memory."""
    from rich.table import Table

    # First, find the memory
    memory = await store.get_memory_by_prefix(memory_id)
    if not memory:
        error(f"No memory found with ID: {memory_id}")
        raise typer.Exit(1)

    # Get the supersession chain
    chain = await store.get_supersession_chain(memory.id)

    if not chain:
        dim("No supersession history for this memory")
        console.print(f"\nCurrent: {memory.content[:100]}")
        return

    table = Table(title=f"Supersession Chain for {memory.id[:8]}")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Created", style="dim")
    table.add_column("Archived", style="yellow")
    table.add_column("Reason", style="cyan")
    table.add_column("Content", style="white", max_width=50)

    for entry in chain:
        content = (
            entry.content[:50] + "..." if len(entry.content) > 50 else entry.content
        )
        content = content.replace("\n", " ")

        archived_at = (
            entry.archived_at.strftime("%Y-%m-%d") if entry.archived_at else "-"
        )

        table.add_row(
            entry.id[:8],
            entry.created_at.strftime("%Y-%m-%d") if entry.created_at else "-",
            archived_at,
            entry.archive_reason or "-",
            content,
        )

    # Add current memory at the end
    current_content = (
        memory.content[:50] + "..." if len(memory.content) > 50 else memory.content
    )
    current_content = current_content.replace("\n", " ")
    table.add_row(
        memory.id[:8],
        memory.created_at.strftime("%Y-%m-%d") if memory.created_at else "-",
        "[green]current[/green]",
        "-",
        f"[green]{current_content}[/green]",
    )

    console.print(table)
    dim(f"\n{len(chain)} superseded entries")
