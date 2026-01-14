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
            typer.Argument(help="Action: list, add, remove, clear"),
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

    config = get_config(config_path)
    database = await get_database(config)

    try:
        async with database.session() as session:
            # Validate scope option
            if scope and scope not in ("personal", "shared", "global"):
                error("--scope must be: personal, shared, or global")
                raise typer.Exit(1)

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

            else:
                error(f"Unknown action: {action}")
                console.print("Valid actions: list, add, remove, clear")
                raise typer.Exit(1)

    finally:
        await database.disconnect()


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
    from sqlalchemy import select

    from ash.db.models import Memory as MemoryModel

    stmt = select(MemoryModel).order_by(MemoryModel.created_at.desc()).limit(limit)

    # Filter by content if query provided
    if query:
        stmt = stmt.where(MemoryModel.content.ilike(f"%{query}%"))

    now = datetime.now(UTC)
    if not include_expired:
        stmt = stmt.where(
            (MemoryModel.expires_at.is_(None)) | (MemoryModel.expires_at > now)
        )

    # Apply filters
    if user_id:
        stmt = stmt.where(MemoryModel.owner_user_id == user_id)
    if chat_id:
        stmt = stmt.where(MemoryModel.chat_id == chat_id)
    if scope == "personal":
        stmt = stmt.where(MemoryModel.owner_user_id.isnot(None))
    elif scope == "shared":
        stmt = stmt.where(
            MemoryModel.owner_user_id.is_(None),
            MemoryModel.chat_id.isnot(None),
        )
    elif scope == "global":
        stmt = stmt.where(
            MemoryModel.owner_user_id.is_(None),
            MemoryModel.chat_id.is_(None),
        )

    result = await session.execute(stmt)
    entries = result.scalars().all()

    if not entries:
        if query:
            warning(f"No memories found matching '{query}'")
        else:
            warning("No memory entries found")
        return

    title = f"Memory Search: '{query}'" if query else "Memory Entries"
    table = Table(title=title)
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Scope", style="magenta", max_width=10)
    table.add_column("Created", style="dim")
    table.add_column("Source", style="cyan")
    table.add_column("Expires", style="yellow")
    table.add_column("Content", style="white", max_width=45)

    for entry in entries:
        content = (
            entry.content[:70] + "..." if len(entry.content) > 70 else entry.content
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

        table.add_row(
            entry.id[:8],
            entry_scope,
            entry.created_at.strftime("%Y-%m-%d"),
            entry.source or "[dim]-[/dim]",
            expires,
            content,
        )

    console.print(table)
    dim(f"\nShowing {len(entries)} entries")


async def _memory_add(
    session, content: str, source: str | None, expires_days: int | None
) -> None:
    """Add a memory entry."""
    from ash.memory import MemoryStore

    store = MemoryStore(session)

    expires_at = None
    if expires_days:
        expires_at = datetime.now(UTC) + timedelta(days=expires_days)

    entry = await store.add_memory(
        content=content,
        source=source,
        expires_at=expires_at,
    )
    await session.commit()

    success(f"Added memory entry: {entry.id[:8]}")
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
    from sqlalchemy import delete, select, text

    from ash.db.models import Memory as MemoryModel

    if not entry_id and not all_entries:
        error("--id or --all is required to remove entries")
        raise typer.Exit(1)

    if all_entries:
        # Build filter description for confirmation
        filter_desc = []
        if user_id:
            filter_desc.append(f"user={user_id}")
        if chat_id:
            filter_desc.append(f"chat={chat_id}")
        if scope:
            filter_desc.append(f"scope={scope}")

        scope_msg = f" matching [{', '.join(filter_desc)}]" if filter_desc else ""

        if not force:
            warning(f"This will remove ALL memory entries{scope_msg}.")
            confirm = typer.confirm("Are you sure?")
            if not confirm:
                dim("Cancelled")
                return

        # Build delete statement with filters
        delete_stmt = delete(MemoryModel)
        if user_id:
            delete_stmt = delete_stmt.where(MemoryModel.owner_user_id == user_id)
        if chat_id:
            delete_stmt = delete_stmt.where(MemoryModel.chat_id == chat_id)
        if scope == "personal":
            delete_stmt = delete_stmt.where(MemoryModel.owner_user_id.isnot(None))
        elif scope == "shared":
            delete_stmt = delete_stmt.where(
                MemoryModel.owner_user_id.is_(None),
                MemoryModel.chat_id.isnot(None),
            )
        elif scope == "global":
            delete_stmt = delete_stmt.where(
                MemoryModel.owner_user_id.is_(None),
                MemoryModel.chat_id.is_(None),
            )

        # Clear embeddings (table may not exist)
        if not (user_id or chat_id or scope):
            try:
                await session.execute(text("DELETE FROM memory_embeddings"))
            except Exception:  # noqa: S110
                pass

        result = await session.execute(delete_stmt)
        await session.commit()

        success(f"Removed {result.rowcount} memory entries")
    else:
        # Find entries matching the ID prefix
        stmt = select(MemoryModel).where(MemoryModel.id.startswith(entry_id))
        result = await session.execute(stmt)
        entries = result.scalars().all()

        if not entries:
            error(f"No memory entry found with ID: {entry_id}")
            raise typer.Exit(1)

        if len(entries) > 1:
            error(
                f"Multiple entries match '{entry_id}'. "
                "Please provide a more specific ID."
            )
            for e in entries:
                console.print(f"  - {e.id}")
            raise typer.Exit(1)

        entry = entries[0]

        if not force:
            warning(f"Content: {entry.content[:100]}...")
            confirm = typer.confirm("Remove this entry?")
            if not confirm:
                dim("Cancelled")
                return

        # Delete embedding if exists
        try:
            await session.execute(
                text("DELETE FROM memory_embeddings WHERE memory_id = :id"),
                {"id": entry.id},
            )
        except Exception:  # noqa: S110
            pass

        # Delete the memory entry
        await session.execute(delete(MemoryModel).where(MemoryModel.id == entry.id))
        await session.commit()

        success(f"Removed memory entry: {entry.id[:8]}")


async def _memory_clear(session, force: bool) -> None:
    """Clear all memory entries."""
    from sqlalchemy import delete, text

    from ash.db.models import Memory as MemoryModel

    if not force:
        warning("This will delete ALL memory entries.")
        confirm = typer.confirm("Are you sure?")
        if not confirm:
            dim("Cancelled")
            return

    # Clear embeddings first
    try:
        await session.execute(text("DELETE FROM memory_embeddings"))
    except Exception:  # noqa: S110
        pass

    result = await session.execute(delete(MemoryModel))
    await session.commit()

    success(f"Cleared {result.rowcount} memory entries")
