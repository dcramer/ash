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

    from ash.graph.edges import (
        ABOUT,
        LEARNED_IN,
        STATED_BY,
        SUPERSEDES,
        get_learned_in_chat,
        get_stated_by_person,
        get_subject_person_ids,
        get_superseded_by,
    )
    from ash.store.types import get_assertion

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

    # Subjects (who this memory is about)
    subject_names: list[str] = []
    subject_person_ids = get_subject_person_ids(store.graph, memory.id)
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

    table.add_row(
        "Sensitivity", memory.sensitivity.value if memory.sensitivity else "public"
    )
    table.add_row("Portable", "yes" if memory.portable else "no")

    assertion = get_assertion(memory)
    if assertion:
        table.add_row("Assertion Kind", assertion.assertion_kind.value)
        table.add_row(
            "Assertion Subjects",
            ", ".join(pid[:8] for pid in assertion.subjects) or "-",
        )
        table.add_row(
            "Assertion Speaker",
            assertion.speaker_person_id[:8] if assertion.speaker_person_id else "-",
        )

    source_chat_id = get_learned_in_chat(store.graph, memory.id)
    source_chat = store.graph.chats.get(source_chat_id) if source_chat_id else None
    if source_chat:
        chat_label = (
            f"{source_chat.chat_type or 'unknown'} "
            f"({source_chat.provider}:{source_chat.provider_id})"
        )
        if source_chat.title:
            chat_label = f'{chat_label} "{source_chat.title}"'
        table.add_row("Learned In", f"{chat_label} [{source_chat.id[:8]}]")
        if source_chat.chat_type == "private":
            table.add_row(
                "DM Isolation",
                f"Locked to DM chat {source_chat.provider_id} for contextual retrieval",
            )
    else:
        table.add_row("Learned In", "- (legacy/no provenance edge)")

    stated_by_pid = get_stated_by_person(store.graph, memory.id)
    if stated_by_pid:
        stated_by_person = people_by_id.get(stated_by_pid)
        if not stated_by_person:
            stated_by_person = await store.get_person(stated_by_pid)
        if stated_by_person:
            table.add_row("Stated By", f"{stated_by_person.name} ({stated_by_pid[:8]})")
        else:
            table.add_row("Stated By", stated_by_pid)
    else:
        table.add_row("Stated By", "-")

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

    if memory.superseded_at:
        table.add_row("Superseded", memory.superseded_at.isoformat())
    superseded_by_id = get_superseded_by(store.graph, memory.id)
    if superseded_by_id:
        table.add_row("Superseded By", superseded_by_id)

    outgoing = store.graph.get_outgoing(memory.id)
    edge_counts: dict[str, int] = {}
    for edge in outgoing:
        edge_counts[edge.edge_type] = edge_counts.get(edge.edge_type, 0) + 1
    table.add_row(
        "Outgoing Edges",
        ", ".join(
            f"{edge_type}={edge_counts.get(edge_type, 0)}"
            for edge_type in (ABOUT, STATED_BY, LEARNED_IN, SUPERSEDES)
        ),
    )

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
    from ash.cli.console import create_table

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

    table = create_table(
        f"Supersession Chain for {memory.id[:8]}",
        [
            ("ID", {"style": "dim", "max_width": 8}),
            ("Created", "dim"),
            ("Archived", "yellow"),
            ("Reason", "cyan"),
            ("Content", {"style": "white", "max_width": 50}),
        ],
    )

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
