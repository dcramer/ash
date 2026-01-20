"""Log viewing commands."""

import json
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import typer

from ash.cli.console import console, dim


def register(app: typer.Typer) -> None:
    """Register the logs command."""

    @app.command()
    def logs(
        query: Annotated[
            list[str] | None,
            typer.Argument(help="Text to search for in log messages"),
        ] = None,
        since: Annotated[
            str | None,
            typer.Option(
                "--since",
                "-s",
                help="Time range: 1h, 30m, 1d, or ISO timestamp",
            ),
        ] = None,
        until: Annotated[
            str | None,
            typer.Option(
                "--until",
                "-u",
                help="End time (default: now)",
            ),
        ] = None,
        level: Annotated[
            str | None,
            typer.Option(
                "--level",
                "-l",
                help="Minimum log level: DEBUG, INFO, WARNING, ERROR",
            ),
        ] = None,
        component: Annotated[
            str | None,
            typer.Option(
                "--component",
                "-c",
                help="Filter by component: events, providers, tools, etc.",
            ),
        ] = None,
        limit: Annotated[
            int,
            typer.Option(
                "--limit",
                "-n",
                help="Maximum entries to show",
            ),
        ] = 50,
        follow: Annotated[
            bool,
            typer.Option(
                "--follow",
                "-f",
                help="Follow mode (like tail -f)",
            ),
        ] = False,
        output_json: Annotated[
            bool,
            typer.Option(
                "--json",
                help="Output as JSON",
            ),
        ] = False,
    ) -> None:
        """View and search Ash logs.

        Logs are stored in ~/.ash/logs/ as daily JSONL files.

        Examples:
            ash logs                           # Show recent logs
            ash logs "schedule"                # Search for "schedule"
            ash logs --level ERROR             # Show errors only
            ash logs --since 1h "failed"       # Last hour + search
            ash logs --component events        # Filter by component
            ash logs -f                        # Follow mode
        """
        from ash.config.paths import get_logs_path

        logs_path = get_logs_path()

        # Parse time range
        try:
            since_dt = parse_time(since) if since else None
            until_dt = parse_time(until) if until else None
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1) from None

        # Parse log level
        try:
            level_value = parse_level(level) if level else None
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1) from None

        # Combine query terms
        search_pattern = " ".join(query) if query else None

        if follow:
            _follow_logs(
                logs_path,
                search_pattern=search_pattern,
                level_value=level_value,
                component=component,
                output_json=output_json,
            )
        else:
            entries = query_logs(
                logs_path,
                since=since_dt,
                until=until_dt,
                search_pattern=search_pattern,
                level_value=level_value,
                component=component,
                limit=limit,
            )

            if not entries:
                console.print(dim("No log entries found."))
                return

            _display_entries(entries, output_json)


def query_logs(
    logs_path: Path,
    since: datetime | None = None,
    until: datetime | None = None,
    search_pattern: str | None = None,
    level_value: int | None = None,
    component: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Query log files and return matching entries.

    Args:
        logs_path: Path to logs directory.
        since: Start time for filtering.
        until: End time for filtering.
        search_pattern: Text to search for in messages.
        level_value: Minimum log level (as int).
        component: Component name filter.
        limit: Maximum entries to return.

    Returns:
        List of matching log entries (newest first).
    """
    if not logs_path.exists():
        return []

    # Determine which log files to read
    log_files = sorted(logs_path.glob("*.jsonl"), reverse=True)
    if not log_files:
        return []

    # If since is specified, filter files by date
    if since:
        since_date = since.strftime("%Y-%m-%d")
        log_files = [f for f in log_files if f.stem >= since_date]

    entries: list[dict[str, Any]] = []

    for log_file in log_files:
        file_entries = _read_log_file(
            log_file,
            since=since,
            until=until,
            search_pattern=search_pattern,
            level_value=level_value,
            component=component,
        )
        entries.extend(file_entries)

        # Stop if we have enough entries
        if limit and len(entries) >= limit:
            break

    # Sort by timestamp descending (newest first) and limit
    entries.sort(key=lambda e: e.get("ts", ""), reverse=True)
    return entries[:limit]


def _read_log_file(
    log_file: Path,
    since: datetime | None = None,
    until: datetime | None = None,
    search_pattern: str | None = None,
    level_value: int | None = None,
    component: str | None = None,
) -> list[dict[str, Any]]:
    """Read and filter entries from a single log file."""
    entries = []

    try:
        with log_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Filter by time
                entry_ts = entry.get("ts")
                if entry_ts:
                    try:
                        ts = datetime.fromisoformat(entry_ts.replace("Z", "+00:00"))
                        if since and ts < since:
                            continue
                        if until and ts > until:
                            continue
                    except ValueError:
                        pass

                # Filter by level
                if level_value is not None:
                    entry_level = entry.get("level", "")
                    if LEVEL_ORDER.get(entry_level, 0) < level_value:
                        continue

                # Filter by component
                if component and entry.get("component") != component:
                    continue

                # Filter by search pattern
                if search_pattern:
                    message = entry.get("message", "")
                    if search_pattern.lower() not in message.lower():
                        continue

                entries.append(entry)
    except OSError:
        pass

    return entries


def _follow_logs(
    logs_path: Path,
    search_pattern: str | None = None,
    level_value: int | None = None,
    component: str | None = None,
    output_json: bool = False,
) -> None:
    """Follow log output in real-time."""

    # Start with today's log file
    current_file: Path | None = None
    file_handle = None
    last_pos = 0

    try:
        while True:
            # Determine current log file (may change at midnight)
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            expected_file = logs_path / f"{today}.jsonl"

            # Switch files if needed
            if current_file != expected_file:
                if file_handle:
                    file_handle.close()
                current_file = expected_file
                if current_file.exists():
                    file_handle = current_file.open()
                    # Seek to end to only show new entries
                    file_handle.seek(0, 2)
                    last_pos = file_handle.tell()
                else:
                    file_handle = None
                    last_pos = 0

            # Read new entries
            if file_handle:
                file_handle.seek(last_pos)
                for line in file_handle:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Apply filters
                    if level_value is not None:
                        entry_level = entry.get("level", "")
                        if LEVEL_ORDER.get(entry_level, 0) < level_value:
                            continue

                    if component and entry.get("component") != component:
                        continue

                    if search_pattern:
                        message = entry.get("message", "")
                        if search_pattern.lower() not in message.lower():
                            continue

                    # Display entry
                    _display_entries([entry], output_json)

                last_pos = file_handle.tell()

            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if file_handle:
            file_handle.close()


def _display_entries(entries: list[dict[str, Any]], output_json: bool) -> None:
    """Display log entries to console."""
    level_styles = {
        "DEBUG": "dim",
        "INFO": "cyan",
        "WARNING": "yellow",
        "ERROR": "red bold",
    }

    for entry in entries:
        if output_json:
            console.print(json.dumps(entry))
        else:
            ts = entry.get("ts", "")[:19]  # Truncate to seconds
            level = entry.get("level", "?")
            component = entry.get("component", "?")
            message = entry.get("message", "")

            level_style = level_styles.get(level, "")
            level_display = f"[{level_style}]{level:<7}[/{level_style}]"

            console.print(
                f"[dim]{ts}[/dim] {level_display} [blue]{component:<10}[/blue] {message}"
            )

            # Show exception if present
            if exc := entry.get("exception"):
                console.print(f"[dim]{exc}[/dim]")


# Level order mapping used throughout this module
LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "WARN": 30, "ERROR": 40}


def parse_time(time_str: str) -> datetime:
    """Parse time string to datetime.

    Supports:
    - Relative: 1h, 30m, 1d, 2w
    - ISO: 2026-01-20T10:00:00

    Raises:
        ValueError: If time format is invalid.
    """
    # Try relative time
    match = re.match(r"^(\d+)([mhdw])$", time_str)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        deltas = {
            "m": timedelta(minutes=value),
            "h": timedelta(hours=value),
            "d": timedelta(days=value),
            "w": timedelta(weeks=value),
        }
        return datetime.now(UTC) - deltas[unit]

    # Try ISO format
    try:
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError as e:
        raise ValueError(f"Invalid time format: {time_str}") from e


def parse_level(level_str: str) -> int:
    """Parse log level to integer value.

    Raises:
        ValueError: If level is invalid.
    """
    level_upper = level_str.upper()
    if level_upper not in LEVEL_ORDER:
        raise ValueError(f"Invalid level: {level_str}")
    return LEVEL_ORDER[level_upper]
