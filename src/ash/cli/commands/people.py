"""People management commands."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import click
import typer

from ash.cli.console import console, create_table, dim, error, success, warning

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.graph.store import GraphStore
    from ash.people.types import PersonEntry

logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
    """Register the people command."""

    @app.command()
    def people(
        action: Annotated[
            str | None,
            typer.Argument(
                help="Action: list, show, search, merge, delete, doctor",
            ),
        ] = None,
        target: Annotated[
            str | None,
            typer.Argument(
                help="Person ID or search query",
            ),
        ] = None,
        target2: Annotated[
            str | None,
            typer.Argument(
                help="Second person ID (for merge)",
            ),
        ] = None,
        config_path: Annotated[
            Path | None,
            typer.Option(
                "--config",
                "-c",
                help="Path to configuration file",
            ),
        ] = None,
        auto: Annotated[
            bool,
            typer.Option(
                "--auto",
                help="Auto-merge without confirmation (for doctor)",
            ),
        ] = False,
        force: Annotated[
            bool,
            typer.Option(
                "--force",
                "-f",
                help="Force action without confirmation",
            ),
        ] = False,
    ) -> None:
        """Manage person records.

        Examples:
            ash people list              # List all people
            ash people show <id>         # Show detailed person info
            ash people search <query>    # Search by name/alias
            ash people merge <id1> <id2> # Merge two people
            ash people delete <id>       # Delete a person record
            ash people doctor            # Find and merge duplicates
            ash people doctor --auto     # Auto-merge without confirmation
        """
        if action is None:
            ctx = click.get_current_context()
            click.echo(ctx.get_help())
            raise typer.Exit(0)

        try:
            asyncio.run(
                _run_people_action(
                    action=action,
                    target=target,
                    target2=target2,
                    config_path=config_path,
                    auto=auto,
                    force=force,
                )
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled[/dim]")


async def _with_graph_store(
    config: AshConfig,
    action_name: str,
    callback: Callable[[GraphStore], Awaitable[None]],
) -> None:
    """Run a callback with a GraphStore, handling database lifecycle."""
    from ash.cli.context import get_database

    database = await get_database(config)
    try:
        async with database.session() as session:
            graph_store = await _create_graph_store(config, session)
            if not graph_store:
                error(f"{action_name} requires [embeddings] configuration")
                raise typer.Exit(1)
            await callback(graph_store)
    finally:
        await database.disconnect()


async def _run_people_action(
    action: str,
    target: str | None,
    target2: str | None,
    config_path: Path | None,
    auto: bool,
    force: bool,
) -> None:
    """Run people action asynchronously."""
    from ash.cli.context import get_config

    config = get_config(config_path)

    if action == "list":
        await _people_list()
    elif action == "show":
        if not target:
            error("Usage: ash people show <id>")
            raise typer.Exit(1)
        await _with_graph_store(config, "Show", lambda gs: _people_show(gs, target))
    elif action == "search":
        if not target:
            error("Usage: ash people search <query>")
            raise typer.Exit(1)
        await _people_search(target)
    elif action == "merge":
        if not target or not target2:
            error("Usage: ash people merge <id1> <id2>")
            raise typer.Exit(1)
        await _with_graph_store(
            config, "Merge", lambda gs: _people_merge(gs, target, target2, force)
        )
    elif action == "delete":
        if not target:
            error("Usage: ash people delete <id>")
            raise typer.Exit(1)
        await _with_graph_store(
            config, "Delete", lambda gs: _people_delete(gs, target, force)
        )
    elif action == "doctor":
        await _people_doctor(config, auto)
    else:
        error(f"Unknown action: {action}")
        console.print("Valid actions: list, show, search, merge, delete, doctor")
        raise typer.Exit(1)


async def _people_list() -> None:
    """List all non-merged people."""
    from ash.cli.commands.memory._helpers import get_all_people

    people = await get_all_people()

    if not people:
        console.print("[dim]No people found.[/dim]")
        return

    table = create_table(
        f"People ({len(people)})",
        [
            ("ID", {"style": "dim", "max_width": 12}),
            ("Name", "bold"),
            ("Aliases", ""),
            ("Relationships", ""),
            ("Created By", "cyan"),
            ("Created", "dim"),
        ],
    )

    for person in people:
        aliases = ", ".join(a.value for a in person.aliases) if person.aliases else "-"
        rels = (
            ", ".join(r.relationship for r in person.relationships)
            if person.relationships
            else "-"
        )
        created_by = person.created_by or "-"
        created = person.created_at.strftime("%Y-%m-%d") if person.created_at else "-"
        table.add_row(person.id[:12], person.name, aliases, rels, created_by, created)

    console.print(table)


async def _resolve_person(graph_store: GraphStore, person_id: str) -> PersonEntry:
    """Resolve a person by exact ID or unique prefix match.

    Raises typer.Exit(1) on ambiguous or missing matches.
    """
    people = await graph_store.get_all_people()

    # Exact match
    for p in people:
        if p.id == person_id:
            return p

    # Prefix match
    matches = [p for p in people if p.id.startswith(person_id)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        error(f"Ambiguous ID prefix: {person_id} matches {len(matches)} people")
        raise typer.Exit(1)

    error(f"No person found with ID: {person_id}")
    raise typer.Exit(1)


async def _people_show(graph_store: GraphStore, person_id: str) -> None:
    """Show detailed person info with connected memories and user links."""
    from rich.panel import Panel
    from rich.table import Table

    from ash.graph.types import EdgeType

    person = await _resolve_person(graph_store, person_id)

    # Build details table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("ID", person.id)
    table.add_row("Name", person.name or "-")
    table.add_row("Created By", person.created_by or "-")
    if person.created_at:
        table.add_row("Created", person.created_at.isoformat())
    if person.updated_at:
        table.add_row("Updated", person.updated_at.isoformat())

    if person.aliases:
        alias_str = ", ".join(
            f"{a.value} (from {a.added_by})" if a.added_by else a.value
            for a in person.aliases
        )
        table.add_row("Aliases", alias_str)
    else:
        table.add_row("Aliases", "-")

    if person.relationships:
        for rc in person.relationships:
            stated = f" (stated by {rc.stated_by})" if rc.stated_by else ""
            table.add_row("Relationship", f"{rc.relationship}{stated}")
    else:
        table.add_row("Relationships", "-")

    if person.merged_into:
        table.add_row("Merged Into", person.merged_into)

    # Connected data from graph
    graph = await graph_store.get_graph()

    # Memories about this person
    memory_ids = graph.memories_about(person.id)
    table.add_row("Memories", str(len(memory_ids)))

    # User links (IS_PERSON edges pointing to this person)
    user_node_ids = graph.neighbors(person.id, EdgeType.IS_PERSON, "incoming")
    if user_node_ids:
        users = await graph_store.list_users()
        users_by_id = {u.id: u for u in users}
        user_strs = []
        for uid in user_node_ids:
            user = users_by_id.get(uid)
            if user:
                user_strs.append(
                    f"@{user.username or user.provider_id} ({user.provider})"
                )
            else:
                user_strs.append(uid[:12])
        table.add_row("User Links", ", ".join(user_strs))
    else:
        table.add_row("User Links", "-")

    console.print(Panel(table, title=f"Person: {person.name or person.id[:12]}"))


async def _people_search(query: str) -> None:
    """Search people by name or alias."""
    from ash.cli.commands.memory._helpers import get_all_people

    people = await get_all_people()

    if not people:
        console.print("[dim]No people found.[/dim]")
        return

    query_lower = query.lower()
    matches = []
    for person in people:
        if query_lower in (person.name or "").lower():
            matches.append(person)
            continue
        if any(query_lower in a.value.lower() for a in person.aliases):
            matches.append(person)
            continue
        if any(query_lower in r.relationship.lower() for r in person.relationships):
            matches.append(person)

    if not matches:
        console.print(f"[dim]No people matching '{query}'[/dim]")
        return

    table = create_table(
        f"Search Results ({len(matches)})",
        [
            ("ID", {"style": "dim", "max_width": 12}),
            ("Name", "bold"),
            ("Aliases", ""),
            ("Relationships", ""),
        ],
    )

    for person in matches:
        aliases = ", ".join(a.value for a in person.aliases) if person.aliases else "-"
        rels = (
            ", ".join(r.relationship for r in person.relationships)
            if person.relationships
            else "-"
        )
        table.add_row(person.id[:12], person.name, aliases, rels)

    console.print(table)


async def _people_merge(
    graph_store: GraphStore,
    id1: str,
    id2: str,
    force: bool,
) -> None:
    """Merge two person records."""

    primary = await _resolve_person(graph_store, id1)
    secondary = await _resolve_person(graph_store, id2)

    if primary.id == secondary.id:
        error("Cannot merge a person with themselves")
        raise typer.Exit(1)

    if not force:
        warning(
            f"Merge: {primary.name} ({primary.id[:8]}) <- {secondary.name} ({secondary.id[:8]})"
        )
        dim("The second person will be merged into the first (primary).")
        if not typer.confirm("Proceed?"):
            dim("Cancelled")
            return

    result = await graph_store.merge_people(primary.id, secondary.id)
    if result:
        success(f"Merged {secondary.name} into {primary.name}")
    else:
        error("Merge failed")


async def _people_delete(
    graph_store: GraphStore,
    person_id: str,
    force: bool,
) -> None:
    """Delete a person record."""

    person = await _resolve_person(graph_store, person_id)

    # Check for connected memories
    graph = await graph_store.get_graph()
    memory_ids = graph.memories_about(person.id)

    if not force:
        warning(f"Delete person: {person.name} ({person.id[:8]})")
        if memory_ids:
            warning(f"  {len(memory_ids)} memories reference this person")
            dim(
                "  Memories will be archived (use 'ash memory forget' for explicit control)"
            )
        if not typer.confirm("Proceed?"):
            dim("Cancelled")
            return

    # Archive connected memories first
    if memory_ids:
        await graph_store.forget_person(person.id, delete_person_record=True)
        success(
            f"Deleted {person.name} and archived {len(memory_ids)} connected memories"
        )
    else:
        await graph_store.delete_person(person.id)
        success(f"Deleted {person.name}")


async def _create_graph_store(config: AshConfig, session: object) -> GraphStore | None:
    """Create a GraphStore for commands that need it."""
    from ash.cli.commands.memory._helpers import get_graph_store

    return await get_graph_store(config, session)  # type: ignore[arg-type]


async def _people_doctor(config: AshConfig, auto: bool) -> None:
    """Find and merge duplicate people."""
    from ash.cli.context import get_database

    database = await get_database(config)
    try:
        async with database.session() as session:
            graph_store = await _create_graph_store(config, session)
            if not graph_store:
                error("Doctor requires [embeddings] configuration")
                raise typer.Exit(1) from None

            people = await graph_store.list_people()
            if len(people) < 2:
                console.print(
                    "[dim]Not enough people for dedup (need at least 2).[/dim]"
                )
                return

            # Set up LLM for verification
            try:
                from ash.llm import create_llm_provider

                extraction_model_alias = config.memory.extraction_model or "default"
                model_config = config.get_model(extraction_model_alias)
                api_key = config.resolve_api_key(extraction_model_alias)
                llm = create_llm_provider(
                    model_config.provider,
                    api_key=api_key.get_secret_value() if api_key else None,
                )
                graph_store.set_llm(llm, model_config.model)
            except Exception as e:
                error(f"Failed to initialize LLM for verification: {e}")
                console.print(
                    "[dim]Doctor requires an LLM to verify merge candidates.[/dim]"
                )
                raise typer.Exit(1) from None

            await _run_doctor_scan(graph_store, people, auto)
    finally:
        await database.disconnect()


async def _run_doctor_scan(
    graph_store: GraphStore,
    people: list[PersonEntry],
    auto: bool,
) -> None:
    """Run the doctor scan and merge process."""
    all_ids = [p.id for p in people]

    console.print(f"[dim]Scanning {len(people)} people for duplicates...[/dim]")

    candidates = await graph_store.find_dedup_candidates(all_ids)

    if not candidates:
        success("No duplicates found.")
        return

    console.print(f"\nFound [bold]{len(candidates)}[/bold] merge candidate(s):\n")

    merged = 0
    for primary_id, secondary_id in candidates:
        primary = await graph_store.get_person(primary_id)
        secondary = await graph_store.get_person(secondary_id)
        if not primary or not secondary:
            continue

        console.print(
            f"  [bold]{primary.name}[/bold] (primary) <- [bold]{secondary.name}[/bold] (secondary)"
        )

        should_merge = auto or click.confirm("    Merge these records?", default=True)
        if should_merge:
            result = await graph_store.merge_people(primary_id, secondary_id)
            if result:
                merged += 1
                console.print("    [green]Merged[/green]")
        else:
            console.print("    [dim]Skipped[/dim]")

    if merged:
        success(f"\nMerged {merged} duplicate(s).")
    else:
        console.print("\n[dim]No merges performed.[/dim]")
