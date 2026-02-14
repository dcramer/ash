"""People management commands."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import click
import typer

from ash.cli.console import console, create_table, error, success

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
                help="Action: list, doctor",
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
    ) -> None:
        """Manage person records.

        Examples:
            ash people list      # List all people with aliases/relationships
            ash people doctor    # Find and merge duplicate people
            ash people doctor --auto  # Auto-merge without confirmation
        """
        if action is None:
            ctx = click.get_current_context()
            click.echo(ctx.get_help())
            raise typer.Exit(0)

        try:
            asyncio.run(
                _run_people_action(
                    action=action,
                    config_path=config_path,
                    auto=auto,
                )
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled[/dim]")


async def _run_people_action(
    action: str,
    config_path: Path | None,
    auto: bool,
) -> None:
    """Run people action asynchronously."""
    from ash.cli.context import get_config

    config = get_config(config_path)

    if action == "list":
        await _people_list()
    elif action == "doctor":
        await _people_doctor(config, auto)
    else:
        error(f"Unknown action: {action}")
        console.print("Valid actions: list, doctor")
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
        created = person.created_at.strftime("%Y-%m-%d") if person.created_at else "-"
        table.add_row(person.id[:12], person.name, aliases, rels, created)

    console.print(table)


async def _people_doctor(config: AshConfig, auto: bool) -> None:
    """Find and merge duplicate people."""
    from ash.cli.context import get_database

    database = await get_database(config)
    try:
        async with database.session() as session:
            graph_store = await _create_doctor_graph_store(config, session)
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


async def _create_doctor_graph_store(
    config: AshConfig, session: object
) -> GraphStore | None:
    """Create a GraphStore for the doctor command."""
    from ash.cli.commands.memory._helpers import get_graph_store

    return await get_graph_store(config, session)  # type: ignore[arg-type]


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
