"""List memory entries command."""

from ash.cli.commands.memory._helpers import (
    get_memory_store,
    is_source_self_reference,
)
from ash.cli.console import dim, warning


async def memory_list(
    session,
    limit: int,
    include_expired: bool,
    user_id: str | None,
    chat_id: str | None,
    scope: str | None,
) -> None:
    """List memory entries."""
    from rich.table import Table

    from ash.cli.console import console

    store = get_memory_store()

    # Apply scope-based filters
    owner_user_id = None
    filter_chat_id = None

    if scope == "personal":
        if user_id:
            owner_user_id = user_id
        else:
            # Need to show all personal memories - get all and filter
            pass
    elif scope == "shared":
        if chat_id:
            filter_chat_id = chat_id
    elif user_id:
        owner_user_id = user_id
    elif chat_id:
        filter_chat_id = chat_id

    entries = await store.get_memories(
        limit=limit * 2,  # Get more to filter
        include_expired=include_expired,
        include_superseded=False,
        owner_user_id=owner_user_id,
        chat_id=filter_chat_id,
    )

    # Apply additional filters
    filtered_entries = []
    for entry in entries:
        # Scope filter
        if scope == "personal" and not entry.owner_user_id:
            continue
        if scope == "shared" and entry.owner_user_id:
            continue
        if scope == "global" and (entry.owner_user_id or entry.chat_id):
            continue

        filtered_entries.append(entry)
        if len(filtered_entries) >= limit:
            break

    # Load people for name lookup
    from ash.cli.commands.memory._helpers import get_person_manager

    pm = get_person_manager()
    people = await pm.get_all()
    people_by_id = {p.id: p for p in people}

    if not filtered_entries:
        warning("No memory entries found")
        return

    table = Table(title="Memory Entries")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Type", style="blue", max_width=10)
    table.add_column("About", style="green", max_width=12)
    table.add_column("Source", style="cyan", max_width=12)
    table.add_column("Trust", style="yellow", max_width=8)
    table.add_column("Created", style="dim")
    table.add_column("Content", style="white", max_width=40)

    for entry in filtered_entries:
        content = (
            entry.content[:60] + "..." if len(entry.content) > 60 else entry.content
        )
        content = content.replace("\n", " ")

        # Resolve subject person names
        subject_names = []
        for person_id in entry.subject_person_ids or []:
            person = people_by_id.get(person_id)
            if person:
                subject_names.append(person.name)
            else:
                subject_names.append(person_id[:8])

        # About: subjects if present, otherwise show username for self-referential facts
        if subject_names:
            about = ", ".join(subject_names)
        elif entry.source_username:
            # Self-referential fact - show username to match source column
            about = f"@{entry.source_username}"
        elif entry.source_display_name:
            about = entry.source_display_name
        else:
            about = "[dim]-[/dim]"

        # Source: who provided this information
        if entry.source_username:
            source_display = f"@{entry.source_username}"
        elif entry.source:
            source_display = entry.source
        else:
            source_display = "[dim]-[/dim]"

        # Trust: fact (speaking about self) vs hearsay (speaking about others)
        is_self_ref = is_source_self_reference(
            entry.source_username,
            entry.owner_user_id,
            entry.subject_person_ids,
            people,
            people_by_id,
        )
        if not subject_names or is_self_ref:
            trust = "[green]fact[/green]"
        else:
            trust = "[yellow]hearsay[/yellow]"

        created = entry.created_at.strftime("%Y-%m-%d") if entry.created_at else "-"

        table.add_row(
            entry.id[:8],
            entry.memory_type.value,
            about,
            source_display,
            trust,
            created,
            content,
        )

    console.print(table)
    dim(f"\nShowing {len(filtered_entries)} entries")
