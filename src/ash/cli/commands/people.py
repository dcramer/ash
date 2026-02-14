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
    from ash.db import Database
    from ash.people import PersonManager
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
    from ash.people import create_person_manager

    pm = create_person_manager()
    people = await pm.list_all()

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
    from ash.llm import create_llm_provider
    from ash.people import create_person_manager

    pm = create_person_manager()
    people = await pm.list_all()

    if len(people) < 2:
        console.print("[dim]Not enough people for dedup (need at least 2).[/dim]")
        return

    # Set up LLM for verification (required)
    try:
        extraction_model_alias = config.memory.extraction_model or "default"
        model_config = config.get_model(extraction_model_alias)
        api_key = config.resolve_api_key(extraction_model_alias)
        llm = create_llm_provider(
            model_config.provider,
            api_key=api_key.get_secret_value() if api_key else None,
        )
        pm.set_llm(llm, model_config.model)
    except Exception as e:
        error(f"Failed to initialize LLM for verification: {e}")
        console.print("[dim]Doctor requires an LLM to verify merge candidates.[/dim]")
        raise typer.Exit(1) from None

    # Try to wire memory manager for auto-remap (best-effort).
    # The scan must run inside the DB session scope so remap works during merges.
    database = await _try_init_memory_manager(config, pm)
    try:
        await _run_doctor_scan(pm, people, auto)
    finally:
        if database:
            await database.disconnect()


async def _try_init_memory_manager(
    config: AshConfig, pm: PersonManager
) -> Database | None:
    """Try to wire a memory manager for auto-remap on merge.

    Returns the Database handle (caller must disconnect) so the connection
    stays alive during the doctor scan. Returns None if memory is unavailable.
    """
    from ash.cli.context import get_database

    try:
        from ash.memory import create_memory_manager

        if not config.embeddings:
            return None

        embeddings_key = config.resolve_embeddings_api_key()
        if not embeddings_key:
            return None

        from ash.llm import create_registry

        default_model = config.get_model("default")
        if default_model.provider == "anthropic":
            anthropic_key = config.resolve_api_key("default")
        else:
            anthropic_key = config._resolve_provider_api_key("anthropic")

        llm_registry = create_registry(
            anthropic_api_key=anthropic_key.get_secret_value()
            if anthropic_key
            else None,
            openai_api_key=embeddings_key.get_secret_value()
            if config.embeddings.provider == "openai"
            else None,
        )

        database = await get_database(config)
        # Use session_factory directly to get a long-lived session that
        # stays open during the entire doctor scan (not a context manager
        # that closes on exit).
        session = database.session_factory()
        memory_manager = await create_memory_manager(
            db_session=session,
            llm_registry=llm_registry,
            embedding_model=config.embeddings.model,
            embedding_provider=config.embeddings.provider,
            max_entries=config.memory.max_entries,
            person_manager=pm,
        )
        pm.set_memory_manager(memory_manager)
        return database
    except Exception:
        logger.debug("Failed to initialize memory manager for doctor", exc_info=True)
        return None


async def _run_doctor_scan(
    pm: PersonManager,
    people: list[PersonEntry],
    auto: bool,
) -> None:
    """Run the doctor scan and merge process."""
    all_ids = [p.id for p in people]

    console.print(f"[dim]Scanning {len(people)} people for duplicates...[/dim]")

    candidates = await pm.find_dedup_candidates(all_ids)

    if not candidates:
        success("No duplicates found.")
        return

    console.print(f"\nFound [bold]{len(candidates)}[/bold] merge candidate(s):\n")

    merged = 0
    for primary_id, secondary_id in candidates:
        primary = await pm.get(primary_id)
        secondary = await pm.get(secondary_id)
        if not primary or not secondary:
            continue

        console.print(
            f"  [bold]{primary.name}[/bold] (primary) <- [bold]{secondary.name}[/bold] (secondary)"
        )

        should_merge = auto or click.confirm("    Merge these records?", default=True)
        if should_merge:
            result = await pm.merge(primary_id, secondary_id)
            if result:
                merged += 1
                console.print("    [green]Merged[/green]")
        else:
            console.print("    [dim]Skipped[/dim]")

    if merged:
        success(f"\nMerged {merged} duplicate(s).")
    else:
        console.print("\n[dim]No merges performed.[/dim]")
