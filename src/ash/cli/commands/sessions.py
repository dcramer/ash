"""Session management commands."""

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from ash.cli.console import console, dim, error, success, warning
from ash.cli.context import get_config, get_database


def register(app: typer.Typer) -> None:
    """Register the sessions command."""

    @app.command()
    def sessions(
        action: Annotated[
            str,
            typer.Argument(help="Action: list, search, export, clear"),
        ],
        query: Annotated[
            str | None,
            typer.Option(
                "--query",
                "-q",
                help="Search query for messages",
            ),
        ] = None,
        output: Annotated[
            Path | None,
            typer.Option(
                "--output",
                "-o",
                help="Output file for export",
            ),
        ] = None,
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
    ) -> None:
        """Manage conversation sessions and messages.

        Examples:
            ash sessions list                  # List recent sessions
            ash sessions search -q "hello"     # Search messages
            ash sessions export -o backup.json # Export all sessions
            ash sessions clear                 # Clear all history
        """
        try:
            asyncio.run(
                _run_sessions_action(
                    action=action,
                    query=query,
                    output=output,
                    limit=limit,
                    config_path=config_path,
                    force=force,
                )
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled[/dim]")


async def _run_sessions_action(
    action: str,
    query: str | None,
    output: Path | None,
    limit: int,
    config_path: Path | None,
    force: bool,
) -> None:
    """Run sessions action asynchronously."""
    config = get_config(config_path)
    database = await get_database(config)

    try:
        async with database.session() as session:
            if action == "list":
                await _sessions_list(session, limit)

            elif action == "search":
                if not query:
                    error("--query is required for search")
                    raise typer.Exit(1)
                await _sessions_search(session, query, limit)

            elif action == "export":
                await _sessions_export(session, output)

            elif action == "clear":
                await _sessions_clear(session, force)

            else:
                error(f"Unknown action: {action}")
                console.print("Valid actions: list, search, export, clear")
                raise typer.Exit(1)

    finally:
        await database.disconnect()


async def _sessions_list(session, limit: int) -> None:
    """List conversation sessions."""
    from rich.table import Table
    from sqlalchemy import func, select

    from ash.db.models import Message
    from ash.db.models import Session as DbSession

    # Get sessions with message counts
    stmt = (
        select(
            DbSession,
            func.count(Message.id).label("message_count"),
        )
        .outerjoin(Message)
        .group_by(DbSession.id)
        .order_by(DbSession.updated_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    rows = result.all()

    if not rows:
        warning("No sessions found")
        return

    table = Table(title="Conversation Sessions")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Provider", style="cyan")
    table.add_column("Chat ID", style="dim", max_width=15)
    table.add_column("Messages", style="green", justify="right")
    table.add_column("Last Updated", style="dim")

    for sess, msg_count in rows:
        table.add_row(
            sess.id[:8],
            sess.provider,
            sess.chat_id[:15] if len(sess.chat_id) > 15 else sess.chat_id,
            str(msg_count),
            sess.updated_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)
    dim(f"\nShowing {len(rows)} sessions")


async def _sessions_search(session, query: str, limit: int) -> None:
    """Search messages."""
    from rich.table import Table
    from sqlalchemy import select

    from ash.db.models import Message
    from ash.db.models import Session as DbSession

    stmt = (
        select(Message)
        .join(DbSession)
        .where(Message.content.ilike(f"%{query}%"))
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    messages = result.scalars().all()

    if not messages:
        warning(f"No messages found matching '{query}'")
        return

    table = Table(title=f"Message Search: '{query}'")
    table.add_column("Time", style="dim")
    table.add_column("Role", style="cyan")
    table.add_column("Content", style="white", max_width=60)

    for msg in messages:
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        content = content.replace("\n", " ")
        table.add_row(
            msg.created_at.strftime("%Y-%m-%d %H:%M"),
            msg.role,
            content,
        )

    console.print(table)


async def _sessions_export(session, output: Path | None) -> None:
    """Export all sessions and messages."""
    from sqlalchemy import select

    from ash.db.models import Message
    from ash.db.models import Session as DbSession

    sessions_result = await session.execute(
        select(DbSession).order_by(DbSession.created_at)
    )
    db_sessions = sessions_result.scalars().all()

    export_data = []
    for sess in db_sessions:
        messages_result = await session.execute(
            select(Message)
            .where(Message.session_id == sess.id)
            .order_by(Message.created_at)
        )
        messages = messages_result.scalars().all()

        export_data.append(
            {
                "session_id": sess.id,
                "provider": sess.provider,
                "chat_id": sess.chat_id,
                "user_id": sess.user_id,
                "created_at": sess.created_at.isoformat(),
                "updated_at": sess.updated_at.isoformat(),
                "messages": [
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "created_at": msg.created_at.isoformat(),
                    }
                    for msg in messages
                ],
            }
        )

    json_output = json.dumps(export_data, indent=2)

    if output:
        output.write_text(json_output)
        success(f"Exported {len(export_data)} sessions to {output}")
    else:
        console.print(json_output)


async def _sessions_clear(session, force: bool) -> None:
    """Clear all conversation history."""
    from sqlalchemy import delete, text

    from ash.db.models import Message, ToolExecution
    from ash.db.models import Session as DbSession

    if not force:
        warning("This will delete ALL conversation history.")
        confirm = typer.confirm("Are you sure?")
        if not confirm:
            dim("Cancelled")
            return

    # Clear message embeddings first (table may not exist)
    try:
        await session.execute(text("DELETE FROM message_embeddings"))
    except Exception:  # noqa: S110
        pass

    # Delete in order due to foreign keys
    await session.execute(delete(ToolExecution))
    await session.execute(delete(Message))
    await session.execute(delete(DbSession))
    await session.commit()

    success("All conversation history cleared")
