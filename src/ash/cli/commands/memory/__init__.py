"""Memory management commands."""

import asyncio
from pathlib import Path
from typing import Annotated

import click
import typer

from ash.cli.console import console, error
from ash.cli.context import get_config


def register(app: typer.Typer) -> None:
    """Register the memory command."""

    @app.command()
    def memory(
        action: Annotated[
            str | None,
            typer.Argument(
                help="Action: list, search, show, add, remove, clear, gc, compact, rebuild-index, history, doctor, forget"
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
        delete_person: Annotated[
            bool,
            typer.Option(
                "--delete-person",
                help="Also delete the person record (for forget action)",
            ),
        ] = False,
    ) -> None:
        """Manage memory entries.

        Examples:
            ash memory list                    # List all memories
            ash memory list --scope personal   # List personal memories only
            ash memory list --scope shared     # List shared/group memories
            ash memory list --user bob         # List memories owned by bob
            ash memory search "api keys"       # Semantic search across memories
            ash memory search -q "api keys"    # Same, using --query flag
            ash memory show <id>               # Show full details of a memory
            ash memory add -q "User prefers dark mode"
            ash memory remove <id>             # Remove specific entry
            ash memory remove --all            # Remove all entries
            ash memory clear                   # Clear all memory entries
            ash memory gc                      # Garbage collect expired/superseded
            ash memory rebuild-index           # Rebuild vector index from JSONL
            ash memory history <id>            # Show supersession chain
            ash memory forget <person-id>      # Archive all memories about a person
        """
        if action is None:
            ctx = click.get_current_context()
            click.echo(ctx.get_help())
            raise typer.Exit(0)

        try:
            asyncio.run(
                _run_memory_action(
                    action=action,
                    query=query,
                    entry_id=target,
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
                    delete_person=delete_person,
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
    delete_person: bool = False,
) -> None:
    """Run memory action asynchronously."""
    from ash.cli.commands.memory._helpers import get_store
    from ash.cli.commands.memory.forget import memory_forget
    from ash.cli.commands.memory.list import memory_list
    from ash.cli.commands.memory.maintenance import (
        memory_compact,
        memory_gc,
        memory_rebuild_index,
    )
    from ash.cli.commands.memory.mutate import memory_add, memory_clear, memory_remove
    from ash.cli.commands.memory.search import memory_search
    from ash.cli.commands.memory.show import memory_history, memory_show

    if scope and scope not in ("personal", "shared", "global"):
        error("--scope must be: personal, shared, or global")
        raise typer.Exit(1)

    config = get_config(config_path)
    store = await get_store(config)

    if action == "list":
        if not store:
            error("Memory list requires [embeddings] configuration")
            raise typer.Exit(1)
        await memory_list(store, limit, include_expired, user_id, chat_id, scope)
    elif action == "search":
        search_query = query or entry_id  # entry_id holds positional target
        if not search_query:
            error("Usage: ash memory search <query> or ash memory search -q <query>")
            raise typer.Exit(1)
        if not store:
            error("Semantic search requires [embeddings] configuration")
            raise typer.Exit(1)
        await memory_search(store, search_query, limit, user_id, chat_id)
    elif action == "add":
        if not query:
            error("--query/-q is required to specify content to add")
            raise typer.Exit(1)
        if not store:
            error("Memory add requires [embeddings] configuration for indexing")
            raise typer.Exit(1)
        await memory_add(store, query, source, expires_days)
    elif action == "remove":
        if not entry_id and not all_entries:
            error("<id> or --all is required to remove entries")
            raise typer.Exit(1)
        if not store:
            error("Memory remove requires [embeddings] configuration")
            raise typer.Exit(1)
        await memory_remove(
            store, entry_id, all_entries, force, user_id, chat_id, scope
        )
    elif action == "clear":
        if not store:
            error("Memory clear requires [embeddings] configuration")
            raise typer.Exit(1)
        await memory_clear(store, force)
    elif action == "gc":
        if not store:
            error("Memory gc requires [embeddings] configuration")
            raise typer.Exit(1)
        await memory_gc(store)
    elif action == "rebuild-index":
        if not store:
            error("Rebuild-index requires [embeddings] configuration")
            raise typer.Exit(1)
        await memory_rebuild_index(store)
    elif action == "show":
        if not entry_id:
            error("Usage: ash memory show <id>")
            raise typer.Exit(1)
        if not store:
            error("Memory show requires [embeddings] configuration")
            raise typer.Exit(1)
        await memory_show(store, entry_id)
    elif action == "history":
        if not entry_id:
            error("Usage: ash memory history <id>")
            raise typer.Exit(1)
        if not store:
            error("Memory history requires [embeddings] configuration")
            raise typer.Exit(1)
        await memory_history(store, entry_id)
    elif action == "forget":
        if not entry_id:
            error("Usage: ash memory forget <person-id>")
            raise typer.Exit(1)
        if not store:
            error("Memory forget requires [embeddings] configuration")
            raise typer.Exit(1)
        await memory_forget(store, entry_id, delete_person, force)
    elif action == "compact":
        if not store:
            error("Memory compact requires [embeddings] configuration")
            raise typer.Exit(1)
        await memory_compact(store, force)
    elif action == "doctor":
        from ash.cli.commands.memory.doctor import (
            memory_doctor_attribution,
            memory_doctor_backfill_learned_in,
            memory_doctor_backfill_subjects,
            memory_doctor_contradictions,
            memory_doctor_dedup,
            memory_doctor_embed_missing,
            memory_doctor_fix_names,
            memory_doctor_normalize_semantics,
            memory_doctor_prune_missing_provenance,
            memory_doctor_quality,
            memory_doctor_reclassify,
            memory_doctor_self_facts,
        )

        if not store:
            error("Memory doctor requires [embeddings] configuration")
            raise typer.Exit(1)

        subcommand = entry_id  # positional arg: ash memory doctor <sub>
        if subcommand == "embed-missing":
            await memory_doctor_embed_missing(store, force)
        elif subcommand == "backfill-learned-in":
            await memory_doctor_backfill_learned_in(store, force)
        elif subcommand == "normalize-semantics":
            await memory_doctor_normalize_semantics(store, force)
        elif subcommand == "prune-missing-provenance":
            await memory_doctor_prune_missing_provenance(store, force)
        elif subcommand is None:
            await memory_doctor_backfill_learned_in(store, force)
            await memory_doctor_prune_missing_provenance(store, force)
            await memory_doctor_self_facts(store, force)
            await memory_doctor_backfill_subjects(store, force)
            await memory_doctor_attribution(store, force)
            await memory_doctor_fix_names(store, force)
            await memory_doctor_normalize_semantics(store, force)
            await memory_doctor_reclassify(store, config, force)
            await memory_doctor_quality(store, config, force)
            await memory_doctor_dedup(store, config, force)
            await memory_doctor_contradictions(store, config, force)
        else:
            error(f"Unknown doctor subcommand: {subcommand}")
            console.print(
                "Valid subcommands: embed-missing, backfill-learned-in, "
                "prune-missing-provenance, normalize-semantics "
                "(or omit for full check)"
            )
            raise typer.Exit(1)
    else:
        error(f"Unknown action: {action}")
        console.print(
            "Valid actions: list, search, show, add, remove, clear, gc, compact, rebuild-index, history, forget, doctor"
        )
        raise typer.Exit(1)
