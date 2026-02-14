"""Session management commands."""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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


# --- Helper types for timeline and tool stats ---


@dataclass
class TimelineEntry:
    """A single entry in the timeline with nesting info."""

    entry: Any  # One of the Entry types
    timestamp: datetime
    agent_session_id: str | None = None
    depth: int = 0  # Nesting level for subagents


@dataclass
class ToolStats:
    """Aggregated statistics for a tool."""

    name: str
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_duration_ms: int = 0
    durations: list[int] | None = None

    @property
    def avg_duration_ms(self) -> int:
        if self.calls == 0:
            return 0
        return self.total_duration_ms // self.calls


# --- Helper functions for timeline and agent nesting ---


async def _load_timeline(session_dir: Path) -> list[TimelineEntry]:
    """Load all entries as a chronological timeline with nesting info."""
    from ash.sessions import SessionReader
    from ash.sessions.types import (
        AgentSessionEntry,
        CompactionEntry,
        MessageEntry,
        SessionHeader,
        ToolResultEntry,
        ToolUseEntry,
    )

    reader = SessionReader(session_dir)
    entries = await reader.load_entries()

    # Build agent session lookup
    agent_sessions: dict[str, AgentSessionEntry] = {}
    for entry in entries:
        if isinstance(entry, AgentSessionEntry):
            agent_sessions[entry.id] = entry

    timeline: list[TimelineEntry] = []

    for entry in entries:
        timestamp: datetime | None = None
        agent_session_id: str | None = None
        depth = 0

        if isinstance(entry, SessionHeader):
            timestamp = entry.created_at
        elif isinstance(entry, AgentSessionEntry):
            timestamp = entry.created_at
            # Agent sessions themselves are at the parent level
        elif isinstance(entry, MessageEntry):
            timestamp = entry.created_at
            agent_session_id = entry.agent_session_id
        elif isinstance(entry, ToolUseEntry):
            # Tool uses don't have timestamps, use the message timestamp
            # For now, we'll estimate from nearby entries
            agent_session_id = entry.agent_session_id
        elif isinstance(entry, ToolResultEntry):
            agent_session_id = entry.agent_session_id
        elif isinstance(entry, CompactionEntry):
            timestamp = entry.created_at

        # Calculate depth based on agent session chain
        if agent_session_id and agent_session_id in agent_sessions:
            depth = 1  # Inside a subagent

        # Use a placeholder timestamp for entries without one
        if timestamp is None:
            # Look for timestamp in previous entries
            for prev in reversed(timeline):
                if prev.timestamp:
                    timestamp = prev.timestamp
                    break

        if timestamp is None:
            from datetime import UTC

            timestamp = datetime.now(UTC)

        timeline.append(
            TimelineEntry(
                entry=entry,
                timestamp=timestamp,
                agent_session_id=agent_session_id,
                depth=depth,
            )
        )

    return timeline


def _matches_tool_filters(
    tool_use: Any,
    result: Any | None,
    tool_filter: str | None,
    failed_only: bool,
    slow_threshold_ms: int | None,
) -> bool:
    """Check if a tool call matches the given filters."""
    if tool_filter and tool_use.name != tool_filter:
        return False
    if result:
        if failed_only and result.success:
            return False
        if slow_threshold_ms and (result.duration_ms or 0) < slow_threshold_ms:
            return False
    return True


def _compute_tool_stats_from_lookups(
    lookups: "EntryLookups",
    tool_filter: str | None = None,
    failed_only: bool = False,
    slow_threshold_ms: int | None = None,
) -> dict[str, ToolStats]:
    """Compute aggregated tool statistics from prebuilt lookups."""
    stats: dict[str, ToolStats] = {}

    for tool_use_id, result in lookups.tool_results.items():
        tool_use = lookups.tool_uses.get(tool_use_id)
        if not tool_use:
            continue

        if not _matches_tool_filters(
            tool_use, result, tool_filter, failed_only, slow_threshold_ms
        ):
            continue

        name = tool_use.name
        if name not in stats:
            stats[name] = ToolStats(name=name, durations=[])

        s = stats[name]
        s.calls += 1
        if result.success:
            s.successes += 1
        else:
            s.failures += 1
        if result.duration_ms:
            s.total_duration_ms += result.duration_ms
            if s.durations is not None:
                s.durations.append(result.duration_ms)

    return stats


def _get_tool_calls_from_lookups(
    lookups: "EntryLookups",
    tool_filter: str | None = None,
    failed_only: bool = False,
    slow_threshold_ms: int | None = None,
) -> list[tuple[Any, Any, Any]]:
    """Get filtered tool calls from prebuilt lookups.

    Returns list of (ToolUseEntry, ToolResultEntry|None, AgentSessionEntry|None).
    """
    results: list[tuple[Any, Any, Any]] = []

    for tool_use in lookups.tool_uses.values():
        result = lookups.tool_results.get(tool_use.id)

        if not _matches_tool_filters(
            tool_use, result, tool_filter, failed_only, slow_threshold_ms
        ):
            continue

        agent_session = None
        if tool_use.agent_session_id:
            agent_session = lookups.agent_sessions.get(tool_use.agent_session_id)

        results.append((tool_use, result, agent_session))

    return results


@dataclass
class EntryLookups:
    """Prebuilt lookup dictionaries for session entries."""

    entries: list[Any]
    tool_uses: dict[str, Any]
    tool_results: dict[str, Any]
    agent_sessions: dict[str, Any]


async def _load_entries_with_lookups(session_dir: Path) -> EntryLookups:
    """Load entries and build common lookup dictionaries."""
    from ash.sessions import SessionReader
    from ash.sessions.types import (
        AgentSessionEntry,
        MessageEntry,
        ToolResultEntry,
        ToolUseEntry,
    )

    reader = SessionReader(session_dir)
    entries = await reader.load_entries()

    tool_uses: dict[str, ToolUseEntry] = {}
    tool_results: dict[str, ToolResultEntry] = {}
    agent_sessions: dict[str, AgentSessionEntry] = {}

    for entry in entries:
        if isinstance(entry, ToolUseEntry):
            tool_uses[entry.id] = entry
        elif isinstance(entry, ToolResultEntry):
            tool_results[entry.tool_use_id] = entry
        elif isinstance(entry, AgentSessionEntry):
            agent_sessions[entry.id] = entry

    # Extract tool_use blocks embedded in message content
    for entry in entries:
        if isinstance(entry, MessageEntry) and isinstance(entry.content, list):
            for block in entry.content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_id = block["id"]
                    if tool_id not in tool_uses:
                        tool_uses[tool_id] = ToolUseEntry(
                            id=tool_id,
                            message_id=entry.id,
                            name=block["name"],
                            input=block["input"],
                        )

    return EntryLookups(
        entries=entries,
        tool_uses=tool_uses,
        tool_results=tool_results,
        agent_sessions=agent_sessions,
    )


def _find_session_dir(query: str) -> Path | None:
    """Find a session directory matching the query (fuzzy match on key)."""
    from ash.config.paths import get_sessions_path

    sessions_path = get_sessions_path()
    if not sessions_path.exists():
        error(f"No session found matching '{query}'")
        dim("Use 'ash sessions' to see available sessions")
        return None

    matching_dirs = [
        d for d in sessions_path.iterdir() if d.is_dir() and query in d.name
    ]

    if not matching_dirs:
        error(f"No session found matching '{query}'")
        dim("Use 'ash sessions' to see available sessions")
        return None

    if len(matching_dirs) > 1:
        warning(f"Multiple sessions match '{query}':")
        for d in matching_dirs[:5]:
            console.print(f"  - {d.name}")
        if len(matching_dirs) > 5:
            console.print(f"  ... and {len(matching_dirs) - 5} more")
        dim("Please be more specific")
        return None

    return matching_dirs[0]


def register(app: typer.Typer) -> None:
    """Register the sessions command."""

    @app.command()
    def sessions(
        session_key: Annotated[
            str | None,
            typer.Argument(help="Session key (fuzzy match) or 'search'"),
        ] = None,
        subcommand: Annotated[
            str | None,
            typer.Argument(help="Subcommand: events, tools, or search query"),
        ] = None,
        # Options for list/view
        limit: Annotated[
            int,
            typer.Option(
                "--limit",
                "-n",
                help="Maximum entries to show",
            ),
        ] = 20,
        verbose: Annotated[
            bool,
            typer.Option(
                "--verbose",
                "-v",
                help="Show full content without truncation",
            ),
        ] = False,
        # Options for view
        show_tokens: Annotated[
            bool,
            typer.Option(
                "--show-tokens",
                help="Display token counts",
            ),
        ] = False,
        show_timing: Annotated[
            bool,
            typer.Option(
                "--show-timing",
                help="Display tool durations",
            ),
        ] = False,
        # Options for events
        entry_type: Annotated[
            str | None,
            typer.Option(
                "--type",
                "-t",
                help="Filter by entry type (comma-separated: message,tool_use,etc)",
            ),
        ] = None,
        # Options for tools
        tool_name: Annotated[
            str | None,
            typer.Option(
                "--name",
                help="Filter by tool name",
            ),
        ] = None,
        failed: Annotated[
            bool,
            typer.Option(
                "--failed",
                help="Show only failed tool calls",
            ),
        ] = False,
        slow: Annotated[
            int | None,
            typer.Option(
                "--slow",
                help="Show only calls slower than N ms",
            ),
        ] = None,
        summary: Annotated[
            bool,
            typer.Option(
                "--summary",
                help="Show aggregated stats only (for tools)",
            ),
        ] = False,
        # Output format
        json_output: Annotated[
            bool,
            typer.Option(
                "--json",
                help="Machine-readable JSON output",
            ),
        ] = False,
        # Clear option
        force: Annotated[
            bool,
            typer.Option(
                "--force",
                "-f",
                help="Force action without confirmation",
            ),
        ] = False,
    ) -> None:
        """Manage and debug conversation sessions.

        Sessions are stored as JSONL files in ~/.ash/sessions/.

        Examples:
            ash sessions                           # List recent sessions
            ash sessions telegram_123              # View session (fuzzy match)
            ash sessions telegram_123 events       # Show event timeline
            ash sessions telegram_123 tools        # Show tool analysis
            ash sessions telegram_123 --show-timing   # View with durations
            ash sessions search "hello"            # Search across sessions
            ash sessions clear                     # Clear all history
        """
        try:
            # No args = list sessions
            if session_key is None:
                asyncio.run(_sessions_list(limit))
                return

            # Special keywords
            if session_key == "search":
                if not subcommand:
                    error('Search query required: ash sessions search "query"')
                    raise typer.Exit(1)
                asyncio.run(_sessions_search(subcommand, limit))
                return

            if session_key == "clear":
                _sessions_clear(force)
                return

            # session_key is a session key, check for subcommands
            if subcommand == "events":
                asyncio.run(
                    _sessions_events(
                        session_key,
                        entry_types=entry_type.split(",") if entry_type else None,
                        json_output=json_output,
                        verbose=verbose,
                    )
                )
            elif subcommand == "tools":
                asyncio.run(
                    _sessions_tools(
                        session_key,
                        tool_name=tool_name,
                        failed_only=failed,
                        slow_threshold_ms=slow,
                        summary_only=summary,
                        json_output=json_output,
                        verbose=verbose,
                    )
                )
            else:
                # Default: view session
                asyncio.run(
                    _sessions_view(
                        session_key,
                        verbose=verbose,
                        show_tokens=show_tokens,
                        show_timing=show_timing,
                    )
                )

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


async def _sessions_view(
    query: str,
    verbose: bool,
    show_tokens: bool = False,
    show_timing: bool = False,
) -> None:
    """View a session with full conversation and tool calls."""
    from rich.markdown import Markdown
    from rich.panel import Panel

    from ash.sessions.types import (
        CompactionEntry,
        MessageEntry,
        SessionHeader,
        ToolResultEntry,
    )

    session_dir = _find_session_dir(query)
    if not session_dir:
        return

    lookups = await _load_entries_with_lookups(session_dir)
    entries = lookups.entries
    tool_uses = lookups.tool_uses
    tool_results = lookups.tool_results
    agent_sessions = lookups.agent_sessions

    if not entries:
        warning(f"Session '{session_dir.name}' is empty")
        return

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

            # Add token count if requested
            if show_tokens and entry.token_count:
                header_parts.append(f"[dim]({entry.token_count} tokens)[/dim]")

            header = " ".join(header_parts)

            # Extract content
            if isinstance(entry.content, str):
                content_text = entry.content
                tool_use_blocks = []
            else:
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
                if len(content_text) > 2000 and not verbose:
                    content_text = content_text[:2000] + "\n... [truncated]"
                console.print(Markdown(content_text))

            # Show tool calls for this message
            for tool_block in tool_use_blocks:
                tool_id = tool_block["id"]
                tool_name = tool_block["name"]
                tool_input = tool_block["input"]
                result = tool_results.get(tool_id)

                # Check if this invokes a subagent
                subagent = None
                for agent in agent_sessions.values():
                    if agent.parent_tool_use_id == tool_id:
                        subagent = agent
                        break

                _print_tool_call(
                    tool_name,
                    tool_input,
                    result,
                    verbose,
                    show_timing,
                    subagent,
                    lookups,
                )

            console.print()

        elif isinstance(entry, ToolResultEntry):
            if entry.tool_use_id not in tool_uses:
                status = "[green]✓[/green]" if entry.success else "[red]✗ failed[/red]"
                duration = ""
                if show_timing and entry.duration_ms:
                    duration = f" {_format_duration(entry.duration_ms)}"
                console.print(
                    f"  [bold magenta]🔧 tool call[/bold magenta] {status}{duration}"
                )
                _print_output_lines(entry.output, verbose)
                console.print()

        elif isinstance(entry, CompactionEntry):
            console.print(
                f"[dim italic]--- Context compacted: "
                f"{entry.tokens_before} → {entry.tokens_after} tokens ---[/dim italic]"
            )
            console.print()

    dim(f"\nTotal entries: {len(entries)}")


async def _sessions_events(
    query: str,
    entry_types: list[str] | None = None,
    json_output: bool = False,
    verbose: bool = False,
) -> None:
    """Show all entries chronologically with full metadata."""
    from ash.sessions.types import (
        AgentSessionEntry,
        CompactionEntry,
        MessageEntry,
        SessionHeader,
        ToolResultEntry,
        ToolUseEntry,
    )

    session_dir = _find_session_dir(query)
    if not session_dir:
        return

    timeline = await _load_timeline(session_dir)

    if not timeline:
        warning(f"Session '{session_dir.name}' is empty")
        return

    # Filter by entry types if specified
    if entry_types:
        type_set = set(entry_types)
        timeline = [
            te for te in timeline if getattr(te.entry, "type", None) in type_set
        ]

    if json_output:
        # Output as JSON array
        output = []
        for te in timeline:
            entry_dict = te.entry.to_dict()
            entry_dict["_depth"] = te.depth
            entry_dict["_agent_session_id"] = te.agent_session_id
            output.append(entry_dict)
        console.print(json.dumps(output, indent=2, default=str))
        return

    # Build agent session lookup for display
    agent_sessions: dict[str, AgentSessionEntry] = {}
    for te in timeline:
        if isinstance(te.entry, AgentSessionEntry):
            agent_sessions[te.entry.id] = te.entry

    # Human-readable timeline
    for te in timeline:
        entry = te.entry
        ts = te.timestamp.strftime("%H:%M:%S.%f")[:-3]
        indent = "  │ " * te.depth

        if isinstance(entry, SessionHeader):
            console.print(
                f"[dim]{ts}[/dim] {indent}[bold blue]SESSION[/bold blue]     "
                f"[cyan]{entry.provider}[/cyan]"
            )

        elif isinstance(entry, AgentSessionEntry):
            console.print(
                f"[dim]{ts}[/dim] {indent}[bold yellow]AGENT[/bold yellow]       "
                f"{entry.agent_type}:{entry.agent_name}"
            )

        elif isinstance(entry, MessageEntry):
            role = entry.role.upper()
            role_style = {
                "user": "green",
                "assistant": "cyan",
                "system": "yellow",
            }.get(entry.role, "white")

            tokens = f"{entry.token_count} tokens" if entry.token_count else ""
            text = _extract_message_text(entry.content)
            if len(text) > 60 and not verbose:
                text = text[:60] + "..."
            text = text.replace("\n", " ")

            console.print(
                f"[dim]{ts}[/dim] {indent}[bold {role_style}]MESSAGE[/bold {role_style}]     "
                f'{role:9} [dim]{tokens:>10}[/dim]   "{text}"'
            )

        elif isinstance(entry, ToolUseEntry):
            input_summary = _format_tool_input(entry.name, entry.input, verbose)
            if len(input_summary) > 50 and not verbose:
                input_summary = input_summary[:50] + "..."

            console.print(
                f"[dim]{ts}[/dim] {indent}[bold magenta]TOOL_USE[/bold magenta]    "
                f"{entry.name:12} id={entry.id[:12]}... {input_summary}"
            )

        elif isinstance(entry, ToolResultEntry):
            status = "[green]ok[/green]" if entry.success else "[red]failed[/red]"
            duration = f"{entry.duration_ms}ms" if entry.duration_ms else ""
            output_preview = entry.output[:40].replace("\n", " ") if not verbose else ""
            if len(entry.output) > 40 and not verbose:
                output_preview += "..."

            console.print(
                f"[dim]{ts}[/dim] {indent}[bold magenta]TOOL_RESULT[/bold magenta] "
                f'{duration:>6} {status:8} "{output_preview}"'
            )

        elif isinstance(entry, CompactionEntry):
            console.print(
                f"[dim]{ts}[/dim] {indent}[bold dim]COMPACTION[/bold dim]  "
                f"{entry.tokens_before} → {entry.tokens_after} tokens"
            )

    dim(f"\nTotal entries: {len(timeline)}")


async def _sessions_tools(
    query: str,
    tool_name: str | None = None,
    failed_only: bool = False,
    slow_threshold_ms: int | None = None,
    summary_only: bool = False,
    json_output: bool = False,
    verbose: bool = False,
) -> None:
    """Show tool analysis with filtering and aggregation."""
    from rich.table import Table

    session_dir = _find_session_dir(query)
    if not session_dir:
        return

    lookups = await _load_entries_with_lookups(session_dir)
    stats = _compute_tool_stats_from_lookups(
        lookups,
        tool_filter=tool_name,
        failed_only=failed_only,
        slow_threshold_ms=slow_threshold_ms,
    )

    if not stats:
        warning("No matching tool calls found")
        return

    if json_output:
        if summary_only:
            output = {
                name: {
                    "calls": s.calls,
                    "successes": s.successes,
                    "failures": s.failures,
                    "avg_duration_ms": s.avg_duration_ms,
                }
                for name, s in stats.items()
            }
        else:
            calls = _get_tool_calls_from_lookups(
                lookups,
                tool_filter=tool_name,
                failed_only=failed_only,
                slow_threshold_ms=slow_threshold_ms,
            )
            output = []
            for tool_use, result, agent in calls:
                call_data = {
                    "id": tool_use.id,
                    "name": tool_use.name,
                    "input": tool_use.input,
                }
                if result:
                    call_data["success"] = result.success
                    call_data["duration_ms"] = result.duration_ms
                    call_data["output"] = result.output if verbose else None
                if agent:
                    call_data["agent"] = f"{agent.agent_type}:{agent.agent_name}"
                output.append(call_data)
        console.print(json.dumps(output, indent=2, default=str))
        return

    if summary_only:
        # Summary table
        table = Table(title="Tool Summary")
        table.add_column("Tool", style="magenta")
        table.add_column("Calls", justify="right")
        table.add_column("Success", justify="right", style="green")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Avg Duration", justify="right", style="dim")

        for name, s in sorted(stats.items(), key=lambda x: x[1].calls, reverse=True):
            table.add_row(
                name,
                str(s.calls),
                str(s.successes),
                str(s.failures),
                _format_duration(s.avg_duration_ms) if s.avg_duration_ms else "-",
            )

        console.print(table)
    else:
        # Detailed list
        calls = _get_tool_calls_from_lookups(
            lookups,
            tool_filter=tool_name,
            failed_only=failed_only,
            slow_threshold_ms=slow_threshold_ms,
        )

        for tool_use, result, agent in calls:
            if result:
                status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
                duration = (
                    _format_duration(result.duration_ms) if result.duration_ms else ""
                )
            else:
                status = "[yellow]⏳[/yellow]"
                duration = ""

            agent_info = ""
            if agent:
                agent_info = f" [dim]({agent.agent_type}:{agent.agent_name})[/dim]"

            console.print(
                f"[bold magenta]{tool_use.name}[/bold magenta] {status} {duration}{agent_info}"
            )

            input_summary = _format_tool_input(tool_use.name, tool_use.input, verbose)
            console.print(f"  [dim]{input_summary}[/dim]")

            if verbose and result:
                _print_output_lines(result.output, verbose)

            console.print()

        # Print summary at the end
        dim(f"\nTotal: {len(calls)} tool calls")


def _format_duration(ms: int | None) -> str:
    """Format duration in human-readable form."""
    if ms is None:
        return ""
    if ms < 1000:
        return f"{ms}ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    return f"{minutes:.1f}m"


def _print_output_lines(output_text: str, verbose: bool) -> None:
    """Print output lines with truncation."""
    lines = output_text.strip().split("\n")
    show_all = verbose or len(lines) <= 5
    limit = 50 if show_all else 3

    for line in lines[:limit]:
        console.print(
            f"     [dim]│[/dim] {line[:200] + '...' if len(line) > 200 else line}"
        )

    remaining = len(lines) - limit
    if remaining > 0:
        hint = "" if show_all else ", use -v for full"
        console.print(f"     [dim]│ ... ({remaining} more lines{hint})[/dim]")


def _print_tool_call(
    name: str,
    input_data: dict[str, Any],
    result: Any | None,
    verbose: bool,
    show_timing: bool = False,
    subagent: Any | None = None,
    lookups: EntryLookups | None = None,
) -> None:
    """Print a tool call with its result and optional subagent box."""
    input_summary = _format_tool_input(name, input_data, verbose)

    if result is None:
        status = "[yellow]⏳ pending[/yellow]"
        output_text = None
        duration = ""
    elif result.success:
        status = "[green]✓[/green]"
        output_text = result.output
        duration = (
            f" {_format_duration(result.duration_ms)}"
            if show_timing and result.duration_ms
            else ""
        )
    else:
        status = "[red]✗ failed[/red]"
        output_text = result.output
        duration = (
            f" {_format_duration(result.duration_ms)}"
            if show_timing and result.duration_ms
            else ""
        )

    console.print(f"  [bold magenta]🔧 {name}[/bold magenta] {status}{duration}")

    if input_summary:
        console.print(f"     [dim]{input_summary}[/dim]")

    # If this tool invokes a subagent, show a box with nested content
    if subagent and lookups:
        _print_subagent_box(subagent, lookups, verbose, show_timing)
    elif output_text and not subagent:
        _print_output_lines(output_text, verbose)


def _print_subagent_box(
    agent: Any,
    lookups: EntryLookups,
    verbose: bool,
    show_timing: bool,
) -> None:
    """Print a box showing subagent activity."""
    from ash.sessions.types import MessageEntry, ToolResultEntry, ToolUseEntry

    agent_id = agent.id
    agent_label = f"{agent.agent_type}:{agent.agent_name}"

    # Collect entries belonging to this agent session
    agent_entries = [
        e
        for e in lookups.entries
        if (
            (isinstance(e, MessageEntry) and e.agent_session_id == agent_id)
            or (isinstance(e, ToolUseEntry) and e.agent_session_id == agent_id)
            or (isinstance(e, ToolResultEntry) and e.agent_session_id == agent_id)
        )
    ]

    if not agent_entries:
        console.print(f"     ┌─ {agent_label} ─────────────────────┐")
        console.print("     │ [dim](no entries)[/dim]")
        console.print("     └────────────────────────────────────────┘")
        return

    console.print(f"     ┌─ {agent_label} ────────────────────┐")

    for e in agent_entries:
        if isinstance(e, ToolUseEntry):
            result = lookups.tool_results.get(e.id)
            if result:
                status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
                duration = (
                    f" {_format_duration(result.duration_ms)}"
                    if show_timing and result.duration_ms
                    else ""
                )
            else:
                status = "[yellow]⏳[/yellow]"
                duration = ""
            console.print(f"     │ 🔧 {e.name} {status}{duration}")

    console.print("     └────────────────────────────────────────┘")


def _format_tool_input(name: str, input_data: dict[str, Any], verbose: bool) -> str:
    """Format tool input for display."""
    if name in ("bash", "bash_tool"):
        cmd = input_data.get("command", "")
        return f"$ {cmd[:100] + '...' if not verbose and len(cmd) > 100 else cmd}"
    if name == "read_file":
        return input_data.get("path", "")
    if name == "write_file":
        path = input_data.get("path", "")
        lines = len(input_data.get("content", "").split("\n"))
        return f"{path} ({lines} lines)"
    if name in ("web_search", "recall"):
        return f"query: {input_data.get('query', '')}"
    if name == "web_fetch":
        return input_data.get("url", "")
    if name == "remember":
        facts = input_data.get("facts", [])
        if facts:
            return f"{len(facts)} facts"
        content = input_data.get("content", "")
        return content[:50] + "..." if len(content) > 50 else content
    if name in ("use_agent", "use_skill"):
        agent = input_data.get("agent", "") or input_data.get("skill", "")
        return agent

    if verbose:
        return json.dumps(input_data, indent=2)[:500]
    if input_data:
        key = next(iter(input_data))
        return f"{key}: {str(input_data[key])[:50]}"
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
