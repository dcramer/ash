"""Session management commands."""

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from ash.cli.console import console, dim, error, success, warning


def _extract_message_text(content: str | list) -> str:
    """Extract plain text from message content.

    Args:
        content: Either a string or a list of content blocks.

    Returns:
        Extracted text content.
    """
    if isinstance(content, str):
        return content

    text_parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))
    return "".join(text_parts)


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

        Sessions are stored as JSONL files in ~/.ash/sessions/.

        Examples:
            ash sessions list                  # List recent sessions
            ash sessions search -q "hello"     # Search messages
            ash sessions export -o backup.json # Export all sessions
            ash sessions clear                 # Clear all history
        """
        try:
            if action == "list":
                asyncio.run(_sessions_list(limit))

            elif action == "search":
                if not query:
                    error("--query is required for search")
                    raise typer.Exit(1)
                asyncio.run(_sessions_search(query, limit))

            elif action == "export":
                asyncio.run(_sessions_export(output))

            elif action == "clear":
                _sessions_clear(force)

            else:
                error(f"Unknown action: {action}")
                console.print("Valid actions: list, search, export, clear")
                raise typer.Exit(1)

        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled[/dim]")


async def _sessions_list(limit: int) -> None:
    """List conversation sessions."""
    from rich.table import Table

    from ash.sessions import SessionManager, SessionReader

    sessions = await SessionManager.list_sessions()

    if not sessions:
        warning("No sessions found")
        return

    # Sort by created_at descending and limit
    sessions.sort(key=lambda s: s["created_at"], reverse=True)
    sessions = sessions[:limit]

    table = Table(title="Conversation Sessions")
    table.add_column("Key", style="dim", max_width=20)
    table.add_column("Provider", style="cyan")
    table.add_column("Chat ID", style="dim", max_width=15)
    table.add_column("Messages", style="green", justify="right")
    table.add_column("Created", style="dim")

    for sess in sessions:
        # Count messages in this session
        from ash.config.paths import get_sessions_path

        session_dir = get_sessions_path() / sess["key"]
        reader = SessionReader(session_dir)
        entries = await reader.load_entries()
        from ash.sessions.types import MessageEntry

        message_count = sum(1 for e in entries if isinstance(e, MessageEntry))

        chat_id = sess.get("chat_id") or ""
        if len(chat_id) > 15:
            chat_id = chat_id[:15]

        table.add_row(
            sess["key"][:20],
            sess["provider"],
            chat_id,
            str(message_count),
            sess["created_at"].strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)
    dim(f"\nShowing {len(sessions)} sessions")


async def _sessions_search(query: str, limit: int) -> None:
    """Search messages across all sessions."""
    from rich.table import Table

    from ash.config.paths import get_sessions_path
    from ash.sessions import SessionReader
    from ash.sessions.types import MessageEntry

    sessions_path = get_sessions_path()
    if not sessions_path.exists():
        warning("No sessions found")
        return

    results: list[tuple[str, MessageEntry]] = []

    # Search across all sessions
    for session_dir in sessions_path.iterdir():
        if not session_dir.is_dir():
            continue

        reader = SessionReader(session_dir)
        matches = await reader.search_messages(query, limit=limit)
        for msg in matches:
            results.append((session_dir.name, msg))
            if len(results) >= limit:
                break

        if len(results) >= limit:
            break

    if not results:
        warning(f"No messages found matching '{query}'")
        return

    # Sort by created_at descending
    results.sort(key=lambda x: x[1].created_at, reverse=True)
    results = results[:limit]

    table = Table(title=f"Message Search: '{query}'")
    table.add_column("Session", style="dim", max_width=15)
    table.add_column("Time", style="dim")
    table.add_column("Role", style="cyan")
    table.add_column("Content", style="white", max_width=60)

    for session_key, msg in results:
        content = _extract_message_text(msg.content)
        if len(content) > 100:
            content = content[:100] + "..."
        content = content.replace("\n", " ")

        table.add_row(
            session_key[:15],
            msg.created_at.strftime("%Y-%m-%d %H:%M"),
            msg.role,
            content,
        )

    console.print(table)


async def _sessions_export(output: Path | None) -> None:
    """Export all sessions to JSON."""
    from ash.config.paths import get_sessions_path
    from ash.sessions import SessionReader
    from ash.sessions.types import MessageEntry, SessionHeader

    sessions_path = get_sessions_path()
    if not sessions_path.exists():
        warning("No sessions found")
        return

    export_data = []

    for session_dir in sorted(sessions_path.iterdir()):
        if not session_dir.is_dir():
            continue

        reader = SessionReader(session_dir)
        entries = await reader.load_entries()

        header = None
        messages = []

        for entry in entries:
            if isinstance(entry, SessionHeader):
                header = entry
            elif isinstance(entry, MessageEntry):
                messages.append(
                    {
                        "id": entry.id,
                        "role": entry.role,
                        "content": _extract_message_text(entry.content),
                        "created_at": entry.created_at.isoformat(),
                    }
                )

        if header:
            export_data.append(
                {
                    "session_key": session_dir.name,
                    "session_id": header.id,
                    "provider": header.provider,
                    "chat_id": header.chat_id,
                    "user_id": header.user_id,
                    "created_at": header.created_at.isoformat(),
                    "messages": messages,
                }
            )

    json_output = json.dumps(export_data, indent=2)

    if output:
        output.write_text(json_output)
        success(f"Exported {len(export_data)} sessions to {output}")
    else:
        console.print(json_output)


def _sessions_clear(force: bool) -> None:
    """Clear all conversation history."""
    import shutil

    from ash.config.paths import get_sessions_path

    sessions_path = get_sessions_path()
    if not sessions_path.exists():
        warning("No sessions found")
        return

    # Count sessions
    session_count = sum(1 for d in sessions_path.iterdir() if d.is_dir())

    if session_count == 0:
        warning("No sessions found")
        return

    if not force:
        warning(
            f"This will delete {session_count} session(s) and all conversation history."
        )
        confirm = typer.confirm("Are you sure?")
        if not confirm:
            dim("Cancelled")
            return

    # Delete all session directories
    for session_dir in sessions_path.iterdir():
        if session_dir.is_dir():
            shutil.rmtree(session_dir)

    success(f"Cleared {session_count} session(s)")
