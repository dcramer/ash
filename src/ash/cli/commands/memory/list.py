"""List memory entries command."""

from __future__ import annotations

from ash.cli.commands.memory._helpers import get_memory_store, get_person_manager
from ash.cli.console import console, dim, warning


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

    store = get_memory_store()

    # Apply scope-based filters
    owner_user_id = None
    filter_chat_id = None

    if scope == "personal":
        if user_id:
            owner_user_id = user_id
        # else: no DB filter -- post-filter catches all personal memories below
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
    pm = get_person_manager()
    people = await pm.get_all()
    people_by_id = {p.id: p for p in people}

    # Build username -> person name lookup
    username_to_name: dict[str, str] = {}
    for p in people:
        if p.name:
            username_to_name[p.name.lower()] = p.name
            for alias in p.aliases:
                username_to_name[alias.value.lower()] = p.name

    if not filtered_entries:
        warning("No memory entries found")
        return

    table = Table(title="Memory Entries")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Type", style="blue", max_width=12)
    table.add_column("About", style="green", max_width=16)
    table.add_column("Said by", style="cyan", max_width=14)
    table.add_column("Created", style="dim", max_width=10)
    table.add_column("Content", style="white", max_width=80)

    for entry in filtered_entries:
        content = entry.content.replace("\n", " ")

        # Resolve subject person names
        subject_names: list[str] = []
        for person_id in entry.subject_person_ids or []:
            person = people_by_id.get(person_id)
            subject_names.append(person.name if person else person_id[:8])

        about = ", ".join(subject_names) if subject_names else "[dim]-[/dim]"

        if entry.source_username:
            resolved = username_to_name.get(entry.source_username.lower())
            said_by = resolved or f"@{entry.source_username}"
        else:
            said_by = "[dim]-[/dim]"

        created = entry.created_at.strftime("%Y-%m-%d") if entry.created_at else "-"

        table.add_row(
            entry.id[:8],
            entry.memory_type.value,
            about,
            said_by,
            created,
            content,
        )

    console.print(table)
    dim(f"\nShowing {len(filtered_entries)} entries")
