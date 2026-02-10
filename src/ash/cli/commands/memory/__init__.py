"""Memory management commands."""

import asyncio
from pathlib import Path
from typing import Annotated

import click
import typer

from ash.cli.console import console, error
from ash.cli.context import get_config, get_database


def register(app: typer.Typer) -> None:
    """Register the memory command."""

    @app.command()
    def memory(
        action: Annotated[
            str | None,
            typer.Argument(
                help="Action: list, show, add, remove, clear, gc, rebuild-index, history, doctor"
            ),
        ] = None,
        target: Annotated[
            str | None,
            typer.Argument(
                help="Target ID for show/history commands",
            ),
        ] = None,
        query: Annotated[
            str | None,
            typer.Option(
                "--query",
                "-q",
                help="Search query or content to add",
            ),
        ] = None,
        entry_id: Annotated[
            str | None,
            typer.Option(
                "--id",
                help="Memory entry ID (for remove, deprecated: use positional arg)",
            ),
        ] = None,
        source: Annotated[
            str | None,
            typer.Option(
                "--source",
                "-s",
                help="Source label for new entry",
            ),
        ] = "cli",
        expires_days: Annotated[
            int | None,
            typer.Option(
                "--expires",
                "-e",
                help="Days until expiration (for add)",
            ),
        ] = None,
        include_expired: Annotated[
            bool,
            typer.Option(
                "--include-expired",
                help="Include expired entries",
            ),
        ] = False,
        limit: Annotated[
            int,
            typer.Option(
                "--limit",
                "-n",
                help="Maximum entries to show",
            ),
        ] = 20,
        config_path: Annotated[
            Path | None,
            typer.Option(
                "--config",
                "-c",
                help="Path to configuration file",
            ),
        ] = None,
        force: Annotated[
            bool,
            typer.Option(
                "--force",
                "-f",
                help="Force action without confirmation",
            ),
        ] = False,
        all_entries: Annotated[
            bool,
            typer.Option(
                "--all",
                help="Remove all entries (for remove action)",
            ),
        ] = False,
        user_id: Annotated[
            str | None,
            typer.Option(
                "--user",
                "-u",
                help="Filter by owner user ID",
            ),
        ] = None,
        chat_id: Annotated[
            str | None,
            typer.Option(
                "--chat",
                help="Filter by chat ID",
            ),
        ] = None,
        scope: Annotated[
            str | None,
            typer.Option(
                "--scope",
                help="Filter by scope: personal, shared, or global",
            ),
        ] = None,
        fix_attribution: Annotated[
            bool,
            typer.Option(
                "--fix-attribution",
                help="Fix memories missing source_user_id attribution (doctor action)",
            ),
        ] = False,
    ) -> None:
        """Manage memory entries.

        Examples:
            ash memory list                    # List all memories
            ash memory list -q "api keys"      # Filter memories by content
            ash memory list --scope personal   # List personal memories only
            ash memory list --scope shared     # List shared/group memories
            ash memory list --user bob         # List memories owned by bob
            ash memory show <id>               # Show full details of a memory
            ash memory add -q "User prefers dark mode"
            ash memory remove <id>             # Remove specific entry
            ash memory remove --all            # Remove all entries
            ash memory clear                   # Clear all memory entries
            ash memory gc                      # Garbage collect expired/superseded
            ash memory rebuild-index           # Rebuild vector index from JSONL
            ash memory history <id>            # Show supersession chain
        """
        if action is None:
            ctx = click.get_current_context()
            click.echo(ctx.get_help())
            raise typer.Exit(0)

        # Use target (positional) or entry_id (--id flag) for ID-based commands
        resolved_id = target or entry_id

        try:
            asyncio.run(
                _run_memory_action(
                    action=action,
                    query=query,
                    entry_id=resolved_id,
                    source=source,
                    expires_days=expires_days,
                    include_expired=include_expired,
                    limit=limit,
                    config_path=config_path,
                    force=force,
                    all_entries=all_entries,
                    user_id=user_id,
                    chat_id=chat_id,
                    scope=scope,
                    fix_attribution=fix_attribution,
                )
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled[/dim]")


async def _run_memory_action(
    action: str,
    query: str | None,
    entry_id: str | None,
    source: str | None,
    expires_days: int | None,
    include_expired: bool,
    limit: int,
    config_path: Path | None,
    force: bool,
    all_entries: bool,
    user_id: str | None,
    chat_id: str | None,
    scope: str | None,
    fix_attribution: bool = False,
) -> None:
    """Run memory action asynchronously."""
    from ash.cli.commands.memory.doctor import memory_doctor
    from ash.cli.commands.memory.list import memory_list
    from ash.cli.commands.memory.maintenance import memory_gc, memory_rebuild_index
    from ash.cli.commands.memory.mutate import memory_add, memory_clear, memory_remove
    from ash.cli.commands.memory.show import memory_history, memory_show

    if scope and scope not in ("personal", "shared", "global"):
        error("--scope must be: personal, shared, or global")
        raise typer.Exit(1)

    config = get_config(config_path)
    database = await get_database(config)

    try:
        async with database.session() as session:
            if action == "list":
                await memory_list(
                    session, query, limit, include_expired, user_id, chat_id, scope
                )
            elif action == "add":
                if not query:
                    error("--query is required to specify content to add")
                    raise typer.Exit(1)
                await memory_add(session, query, source, expires_days)
            elif action == "remove":
                await memory_remove(
                    session, entry_id, all_entries, force, user_id, chat_id, scope
                )
            elif action == "clear":
                await memory_clear(session, force)
            elif action == "gc":
                await memory_gc()
            elif action == "rebuild-index":
                await memory_rebuild_index(session)
            elif action == "show":
                if not entry_id:
                    error("Usage: ash memory show <id>")
                    raise typer.Exit(1)
                await memory_show(entry_id)
            elif action == "history":
                if not entry_id:
                    error("Usage: ash memory history <id>")
                    raise typer.Exit(1)
                await memory_history(entry_id)
            elif action == "doctor":
                await memory_doctor(config, force, fix_attribution)
            else:
                error(f"Unknown action: {action}")
                console.print(
                    "Valid actions: list, show, add, remove, clear, gc, rebuild-index, history, doctor"
                )
                raise typer.Exit(1)
    finally:
        await database.disconnect()
