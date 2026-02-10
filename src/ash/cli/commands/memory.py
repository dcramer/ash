"""Memory management commands."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import click
import typer

from ash.cli.console import console, dim, error, success, warning
from ash.cli.context import get_config, get_database


def register(app: typer.Typer) -> None:
    """Register the memory command."""

    @app.command()
    def memory(
        action: Annotated[
            str | None,
            typer.Argument(
                help="Action: list, add, remove, clear, gc, rebuild-index, history"
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
                help="Memory entry ID (for remove)",
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
    ) -> None:
        """Manage memory entries.

        Examples:
            ash memory list                    # List all memories
            ash memory list -q "api keys"      # Filter memories by content
            ash memory list --scope personal   # List personal memories only
            ash memory list --scope shared     # List shared/group memories
            ash memory list --user bob         # List memories owned by bob
            ash memory add -q "User prefers dark mode"
            ash memory remove --id <uuid>      # Remove specific entry
            ash memory remove --all            # Remove all entries
            ash memory clear                   # Clear all memory entries
            ash memory gc                      # Garbage collect expired/superseded
            ash memory rebuild-index           # Rebuild vector index from JSONL
            ash memory history --id <uuid>     # Show supersession chain
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
                    entry_id=entry_id,
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
) -> None:
    """Run memory action asynchronously."""
    if scope and scope not in ("personal", "shared", "global"):
        error("--scope must be: personal, shared, or global")
        raise typer.Exit(1)

    config = get_config(config_path)
    database = await get_database(config)

    try:
        async with database.session() as session:
            if action == "list":
                await _memory_list(
                    session, query, limit, include_expired, user_id, chat_id, scope
                )
            elif action == "add":
                if not query:
                    error("--query is required to specify content to add")
                    raise typer.Exit(1)
                await _memory_add(session, query, source, expires_days)
            elif action == "remove":
                await _memory_remove(
                    session, entry_id, all_entries, force, user_id, chat_id, scope
                )
            elif action == "clear":
                await _memory_clear(session, force)
            elif action == "gc":
                await _memory_gc()
            elif action == "rebuild-index":
                await _memory_rebuild_index(session)
            elif action == "history":
                if not entry_id:
                    error("--id is required for history action")
                    raise typer.Exit(1)
                await _memory_history(entry_id)
            else:
                error(f"Unknown action: {action}")
                console.print(
                    "Valid actions: list, add, remove, clear, gc, rebuild-index, history"
                )
                raise typer.Exit(1)
    finally:
        await database.disconnect()


def _apply_scope_filters(
    stmt, model, user_id: str | None, chat_id: str | None, scope: str | None
):
    """Apply user/chat/scope filters to a query."""
    if user_id:
        stmt = stmt.where(model.owner_user_id == user_id)
    if chat_id:
        stmt = stmt.where(model.chat_id == chat_id)
    if scope == "personal":
        stmt = stmt.where(model.owner_user_id.isnot(None))
    elif scope == "shared":
        stmt = stmt.where(model.owner_user_id.is_(None), model.chat_id.isnot(None))
    elif scope == "global":
        stmt = stmt.where(model.owner_user_id.is_(None), model.chat_id.is_(None))
    return stmt


async def _memory_list(
    session,
    query: str | None,
    limit: int,
    include_expired: bool,
    user_id: str | None,
    chat_id: str | None,
    scope: str | None,
) -> None:
    """List memory entries."""
    from rich.table import Table

    from ash.memory.file_store import FileMemoryStore

    store = FileMemoryStore()

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
    now = datetime.now(UTC)
    filtered_entries = []
    for entry in entries:
        # Scope filter
        if scope == "personal" and not entry.owner_user_id:
            continue
        if scope == "shared" and entry.owner_user_id:
            continue
        if scope == "global" and (entry.owner_user_id or entry.chat_id):
            continue

        # Content filter
        if query and query.lower() not in entry.content.lower():
            continue

        filtered_entries.append(entry)
        if len(filtered_entries) >= limit:
            break

    if not filtered_entries:
        if query:
            warning(f"No memories found matching '{query}'")
        else:
            warning("No memory entries found")
        return

    title = f"Memory Search: '{query}'" if query else "Memory Entries"
    table = Table(title=title)
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Type", style="blue", max_width=10)
    table.add_column("Scope", style="magenta", max_width=10)
    table.add_column("Created", style="dim")
    table.add_column("Source", style="cyan")
    table.add_column("Expires", style="yellow")
    table.add_column("Content", style="white", max_width=40)

    for entry in filtered_entries:
        content = (
            entry.content[:60] + "..." if len(entry.content) > 60 else entry.content
        )
        content = content.replace("\n", " ")

        # Determine scope
        if entry.owner_user_id:
            entry_scope = f"[cyan]{entry.owner_user_id[:8]}[/cyan]"
        elif entry.chat_id:
            entry_scope = f"[yellow]{entry.chat_id[:8]}[/yellow]"
        else:
            entry_scope = "[dim]global[/dim]"

        if entry.expires_at:
            if entry.expires_at < now:
                expires = "[red]expired[/red]"
            else:
                days_left = (entry.expires_at - now).days
                expires = f"{days_left}d"
        else:
            expires = "[dim]never[/dim]"

        created = entry.created_at.strftime("%Y-%m-%d") if entry.created_at else "-"

        table.add_row(
            entry.id[:8],
            entry.memory_type.value,
            entry_scope,
            created,
            entry.source or "[dim]-[/dim]",
            expires,
            content,
        )

    console.print(table)
    dim(f"\nShowing {len(filtered_entries)} entries")


async def _memory_add(
    session, content: str, source: str | None, expires_days: int | None
) -> None:
    """Add a memory entry."""
    from ash.memory.file_store import FileMemoryStore
    from ash.memory.types import MemoryType

    store = FileMemoryStore()

    expires_at = None
    if expires_days:
        expires_at = datetime.now(UTC) + timedelta(days=expires_days)

    entry = await store.add_memory(
        content=content,
        source=source or "cli",
        memory_type=MemoryType.KNOWLEDGE,
        expires_at=expires_at,
    )

    success(f"Added memory entry: {entry.id[:8]}")
    dim(f"Type: {entry.memory_type.value}")
    if expires_at:
        dim(f"Expires: {expires_at.strftime('%Y-%m-%d')}")


async def _memory_remove(
    session,
    entry_id: str | None,
    all_entries: bool,
    force: bool,
    user_id: str | None,
    chat_id: str | None,
    scope: str | None,
) -> None:
    """Remove memory entries."""
    from sqlalchemy import text

    from ash.memory.file_store import FileMemoryStore

    if not entry_id and not all_entries:
        error("--id or --all is required to remove entries")
        raise typer.Exit(1)

    store = FileMemoryStore()

    if all_entries:
        filter_desc = [
            f"{k}={v}"
            for k, v in [("user", user_id), ("chat", chat_id), ("scope", scope)]
            if v
        ]
        scope_msg = f" matching [{', '.join(filter_desc)}]" if filter_desc else ""

        if not force:
            warning(f"This will remove ALL memory entries{scope_msg}.")
            if not typer.confirm("Are you sure?"):
                dim("Cancelled")
                return

        # Get all entries and filter
        entries = await store.get_all_memories()
        to_remove = []

        for entry in entries:
            if scope == "personal" and not entry.owner_user_id:
                continue
            if scope == "shared" and entry.owner_user_id:
                continue
            if scope == "global" and (entry.owner_user_id or entry.chat_id):
                continue
            if user_id and entry.owner_user_id != user_id:
                continue
            if chat_id and entry.chat_id != chat_id:
                continue
            to_remove.append(entry)

        for entry in to_remove:
            await store.delete_memory(entry.id)
            try:
                await session.execute(
                    text("DELETE FROM memory_embeddings WHERE memory_id = :id"),
                    {"id": entry.id},
                )
            except Exception:  # noqa: S110
                pass

        await session.commit()
        success(f"Removed {len(to_remove)} memory entries")
    else:
        # Find entry by ID prefix
        assert entry_id is not None  # Guaranteed by check above
        entry = await store.get_memory_by_prefix(entry_id)

        if not entry:
            error(f"No memory entry found with ID: {entry_id}")
            raise typer.Exit(1)

        if not force:
            warning(f"Content: {entry.content[:100]}...")
            confirm = typer.confirm("Remove this entry?")
            if not confirm:
                dim("Cancelled")
                return

        # Delete the memory entry
        deleted = await store.delete_memory(entry.id)

        if deleted:
            # Delete embedding if exists
            try:
                await session.execute(
                    text("DELETE FROM memory_embeddings WHERE memory_id = :id"),
                    {"id": entry.id},
                )
                await session.commit()
            except Exception:  # noqa: S110
                pass

            success(f"Removed memory entry: {entry.id[:8]}")
        else:
            error(f"Failed to remove memory entry: {entry.id[:8]}")


async def _memory_clear(session, force: bool) -> None:
    """Clear all memory entries."""
    from sqlalchemy import text

    from ash.memory.file_store import FileMemoryStore

    if not force:
        warning("This will delete ALL memory entries.")
        confirm = typer.confirm("Are you sure?")
        if not confirm:
            dim("Cancelled")
            return

    store = FileMemoryStore()
    entries = await store.get_all_memories()

    # Delete all entries
    for entry in entries:
        await store.delete_memory(entry.id)

    # Clear embeddings
    try:
        await session.execute(text("DELETE FROM memory_embeddings"))
        await session.commit()
    except Exception:  # noqa: S110
        pass

    success(f"Cleared {len(entries)} memory entries")


async def _memory_gc() -> None:
    """Garbage collect expired and superseded memories."""
    from ash.memory.file_store import FileMemoryStore

    store = FileMemoryStore()
    result = await store.gc()

    if result.removed_count == 0:
        dim("No memories to clean up")
    else:
        success(f"Archived and removed {result.removed_count} memories")
        if result.archived_ids:
            dim(f"Archived IDs: {', '.join(id[:8] for id in result.archived_ids[:5])}")
            if len(result.archived_ids) > 5:
                dim(f"  ... and {len(result.archived_ids) - 5} more")


async def _memory_rebuild_index(session) -> None:
    """Rebuild vector index from JSONL source of truth."""
    from ash.config.paths import get_memories_jsonl_path
    from ash.memory.index import rebuild_vector_index_from_jsonl

    memories_path = get_memories_jsonl_path()

    if not memories_path.exists():
        warning("No memories.jsonl file found")
        return

    dim(f"Rebuilding index from {memories_path}")
    count = await rebuild_vector_index_from_jsonl(memories_path)

    success(f"Rebuilt index with {count} embeddings")


async def _memory_history(memory_id: str) -> None:
    """Show supersession chain for a memory."""
    from rich.table import Table

    from ash.memory.file_store import FileMemoryStore

    store = FileMemoryStore()

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
