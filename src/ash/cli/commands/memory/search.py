"""Semantic search command for memory entries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ash.cli.console import console, dim, warning

if TYPE_CHECKING:
    from ash.store.store import Store


async def memory_search(
    manager: Store,
    query: str,
    limit: int,
    user_id: str | None,
    chat_id: str | None,
) -> None:
    """Search memories using semantic similarity."""
    from rich.table import Table

    results = await manager.search(
        query=query,
        limit=limit,
        owner_user_id=user_id,
        chat_id=chat_id,
    )

    if not results:
        warning(f"No memories found matching '{query}'")
        return

    # Load people for subject name resolution
    people = await manager.list_people()
    people_by_id = {p.id: p for p in people}

    table = Table(title=f"Memory Search: '{query}'")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Type", style="blue", max_width=10)
    table.add_column("Score", style="yellow", max_width=6)
    table.add_column("About", style="green", max_width=12)
    table.add_column("Content", style="white", max_width=50)

    for result in results:
        content = (
            result.content[:60] + "..." if len(result.content) > 60 else result.content
        )
        content = content.replace("\n", " ")

        # Resolve subject person names from metadata
        subject_names: list[str] = []
        subject_ids = (
            result.metadata.get("subject_person_ids", []) if result.metadata else []
        )
        for person_id in subject_ids:
            person = people_by_id.get(person_id)
            if person:
                subject_names.append(person.name)
            else:
                subject_names.append(person_id[:8])

        about = ", ".join(subject_names) if subject_names else "[dim]-[/dim]"

        memory_type = (
            result.metadata.get("memory_type", "knowledge")
            if result.metadata
            else "knowledge"
        )

        table.add_row(
            result.id[:8],
            memory_type,
            f"{result.similarity:.2f}",
            about,
            content,
        )

    console.print(table)
    dim(f"\nShowing {len(results)} results")
