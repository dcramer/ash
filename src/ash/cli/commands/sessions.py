"""Session management commands."""

import asyncio
import json
from typing import Annotated, Any

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
            typer.Argument(help="Action: list, view, search, clear"),
        ],
        query: Annotated[
            str | None,
            typer.Option(
                "--query",
                "-q",
                help="Search query or session key for view",
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
        verbose: Annotated[
            bool,
            typer.Option(
                "--verbose",
                "-v",
                help="Show full tool outputs (for view)",
            ),
        ] = False,
    ) -> None:
        """Manage conversation sessions and messages.

        Sessions are stored as JSONL files in ~/.ash/sessions/.

        Examples:
            ash sessions list                        # List recent sessions
            ash sessions view -q telegram_-542      # View session (fuzzy match)
            ash sessions view -q telegram_-542 -v   # View with full tool outputs
            ash sessions search -q "hello"           # Search messages
            ash sessions clear                       # Clear all history
        """
        try:
            if action == "list":
                asyncio.run(_sessions_list(limit))

            elif action == "view":
                if not query:
                    error("--query is required for view (session key or partial match)")
                    raise typer.Exit(1)
                asyncio.run(_sessions_view(query, verbose))

            elif action == "search":
                if not query:
                    error("--query is required for search")
                    raise typer.Exit(1)
                asyncio.run(_sessions_search(query, limit))

            elif action == "clear":
                _sessions_clear(force)

            else:
                error(f"Unknown action: {action}")
                console.print("Valid actions: list, view, search, clear")
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


async def _sessions_view(query: str, verbose: bool) -> None:
    """View a session with full conversation and tool calls."""
    from rich.markdown import Markdown
    from rich.panel import Panel

    from ash.config.paths import get_sessions_path
    from ash.sessions import SessionReader
    from ash.sessions.types import (
        CompactionEntry,
        MessageEntry,
        SessionHeader,
        ToolResultEntry,
        ToolUseEntry,
    )

    sessions_path = get_sessions_path()
    if not sessions_path.exists():
        warning("No sessions found")
        return

    # Find matching session (fuzzy match on key)
    matching_dirs = [
        d for d in sessions_path.iterdir() if d.is_dir() and query in d.name
    ]

    if not matching_dirs:
        error(f"No session found matching '{query}'")
        dim("Use 'ash sessions list' to see available sessions")
        return

    if len(matching_dirs) > 1:
        warning(f"Multiple sessions match '{query}':")
        for d in matching_dirs[:5]:
            console.print(f"  - {d.name}")
        if len(matching_dirs) > 5:
            console.print(f"  ... and {len(matching_dirs) - 5} more")
        dim("Please be more specific")
        return

    session_dir = matching_dirs[0]
    reader = SessionReader(session_dir)
    entries = await reader.load_entries()

    if not entries:
        warning(f"Session '{session_dir.name}' is empty")
        return

    # Build lookup for tool uses and results
    tool_uses: dict[str, ToolUseEntry] = {}
    tool_results: dict[str, ToolResultEntry] = {}

    for entry in entries:
        if isinstance(entry, ToolUseEntry):
            tool_uses[entry.id] = entry
        elif isinstance(entry, ToolResultEntry):
            tool_results[entry.tool_use_id] = entry

    # Also extract tool_use blocks from message content
    for entry in entries:
        if isinstance(entry, MessageEntry) and isinstance(entry.content, list):
            for block in entry.content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    # Create a synthetic ToolUseEntry for display
                    tool_id = block["id"]
                    if tool_id not in tool_uses:
                        tool_uses[tool_id] = ToolUseEntry(
                            id=tool_id,
                            message_id=entry.id,
                            name=block["name"],
                            input=block["input"],
                        )

    console.print()
    console.print(
        Panel(f"[bold]Session: {session_dir.name}[/bold]", style="blue", expand=False)
    )
    console.print()

    for entry in entries:
        if isinstance(entry, SessionHeader):
            dim(
                f"[Session created {entry.created_at.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Provider: {entry.provider}]"
            )
            console.print()

        elif isinstance(entry, MessageEntry):
            role = entry.role.upper()
            timestamp = entry.created_at.strftime("%H:%M:%S")

            # Role-based styling
            if entry.role == "user":
                role_style = "bold green"
            elif entry.role == "assistant":
                role_style = "bold cyan"
            else:
                role_style = "bold yellow"

            # Build header
            header_parts = [f"[{role_style}]{role}[/{role_style}]"]
            if entry.username:
                header_parts.append(f"(@{entry.username})")
            header_parts.append(f"[dim]{timestamp}[/dim]")
            header = " ".join(header_parts)

            # Extract content
            if isinstance(entry.content, str):
                content_text = entry.content
                tool_use_blocks = []
            else:
                # Extract text and tool_use blocks
                text_parts = []
                tool_use_blocks = []
                for block in entry.content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tool_use_blocks.append(block)
                content_text = "\n".join(text_parts)

            console.print(header)
            if content_text.strip():
                # Truncate very long content
                if len(content_text) > 2000 and not verbose:
                    content_text = content_text[:2000] + "\n... [truncated]"
                console.print(Markdown(content_text))

            # Show tool calls for this message
            for tool_block in tool_use_blocks:
                tool_id = tool_block["id"]
                tool_name = tool_block["name"]
                tool_input = tool_block["input"]
                result = tool_results.get(tool_id)

                _print_tool_call(tool_name, tool_input, result, verbose)

            console.print()

        elif isinstance(entry, ToolResultEntry):
            # Show orphaned tool results (when tool_use wasn't persisted in message)
            if entry.tool_use_id not in tool_uses:
                status = "[green]✓[/green]" if entry.success else "[red]✗ failed[/red]"
                console.print(f"  [bold magenta]🔧 tool call[/bold magenta] {status}")
                output_lines = entry.output.strip().split("\n")
                if verbose or len(output_lines) <= 5:
                    for line in output_lines[:50]:
                        if len(line) > 200:
                            line = line[:200] + "..."
                        console.print(f"     [dim]│[/dim] {line}")
                    if len(output_lines) > 50:
                        console.print(
                            f"     [dim]│ ... ({len(output_lines) - 50} more lines)[/dim]"
                        )
                else:
                    for line in output_lines[:3]:
                        if len(line) > 200:
                            line = line[:200] + "..."
                        console.print(f"     [dim]│[/dim] {line}")
                    if len(output_lines) > 3:
                        console.print(
                            f"     [dim]│ ... ({len(output_lines) - 3} more lines, "
                            f"use -v for full)[/dim]"
                        )
                console.print()

        elif isinstance(entry, CompactionEntry):
            console.print(
                f"[dim italic]--- Context compacted: "
                f"{entry.tokens_before} → {entry.tokens_after} tokens ---[/dim italic]"
            )
            console.print()

    dim(f"\nTotal entries: {len(entries)}")


def _print_tool_call(
    name: str,
    input_data: dict[str, Any],
    result: Any | None,
    verbose: bool,
) -> None:
    """Print a tool call with its result."""

    # Format tool input summary
    input_summary = _format_tool_input(name, input_data, verbose)

    # Determine result status
    if result is None:
        status = "[yellow]⏳ pending[/yellow]"
        output_text = None
    elif result.success:
        status = "[green]✓[/green]"
        output_text = result.output
    else:
        status = "[red]✗ failed[/red]"
        output_text = result.output

    # Build tool header
    tool_header = f"  [bold magenta]🔧 {name}[/bold magenta] {status}"
    console.print(tool_header)

    # Show input
    if input_summary:
        console.print(f"     [dim]{input_summary}[/dim]")

    # Show output
    if output_text:
        output_lines = output_text.strip().split("\n")
        if verbose or len(output_lines) <= 5:
            # Show full output
            for line in output_lines[:50]:  # Cap at 50 lines even in verbose
                if len(line) > 200:
                    line = line[:200] + "..."
                console.print(f"     [dim]│[/dim] {line}")
            if len(output_lines) > 50:
                console.print(
                    f"     [dim]│ ... ({len(output_lines) - 50} more lines)[/dim]"
                )
        else:
            # Show truncated output
            for line in output_lines[:3]:
                if len(line) > 200:
                    line = line[:200] + "..."
                console.print(f"     [dim]│[/dim] {line}")
            if len(output_lines) > 3:
                console.print(
                    f"     [dim]│ ... ({len(output_lines) - 3} more lines, use -v for full)[/dim]"
                )


def _format_tool_input(name: str, input_data: dict[str, Any], verbose: bool) -> str:
    """Format tool input for display."""
    match name:
        case "bash" | "bash_tool":
            cmd = input_data.get("command", "")
            if not verbose and len(cmd) > 100:
                cmd = cmd[:100] + "..."
            return f"$ {cmd}"
        case "read_file":
            return input_data.get("path", "")
        case "write_file":
            path = input_data.get("path", "")
            content = input_data.get("content", "")
            lines = len(content.split("\n"))
            return f"{path} ({lines} lines)"
        case "web_search":
            return f"query: {input_data.get('query', '')}"
        case "web_fetch":
            return input_data.get("url", "")
        case "recall":
            return f"query: {input_data.get('query', '')}"
        case "remember":
            facts = input_data.get("facts", [])
            if facts:
                return f"{len(facts)} facts"
            content = input_data.get("content", "")
            if len(content) > 50:
                content = content[:50] + "..."
            return content
        case "use_agent":
            return input_data.get("agent", "")
        case _:
            if verbose:
                return json.dumps(input_data, indent=2)[:500]
            # Show first key-value pair
            if input_data:
                key = next(iter(input_data))
                val = str(input_data[key])[:50]
                return f"{key}: {val}"
            return ""


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
