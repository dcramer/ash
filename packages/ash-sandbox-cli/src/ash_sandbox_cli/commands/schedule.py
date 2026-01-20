"""Schedule management commands for sandboxed CLI."""

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="schedule",
    help="Manage scheduled tasks.",
    no_args_is_help=True,
)

SCHEDULE_FILE = Path("/workspace/schedule.jsonl")


def _get_context() -> dict[str, str]:
    """Get routing context from environment variables."""
    return {
        "session_id": os.environ.get("ASH_SESSION_ID", ""),
        "user_id": os.environ.get("ASH_USER_ID", ""),
        "chat_id": os.environ.get("ASH_CHAT_ID", ""),
        "chat_title": os.environ.get("ASH_CHAT_TITLE", ""),
        "provider": os.environ.get("ASH_PROVIDER", ""),
        "username": os.environ.get("ASH_USERNAME", ""),
        "timezone": os.environ.get("ASH_TIMEZONE", "UTC"),
    }


def _require_routing_context() -> dict[str, str]:
    """Get context and validate required fields for response routing."""
    ctx = _get_context()
    if not ctx["provider"] or not ctx["chat_id"]:
        typer.echo(
            "Error: Scheduling requires a provider context (ASH_PROVIDER and ASH_CHAT_ID). "
            "Cannot schedule tasks from CLI.",
            err=True,
        )
        raise typer.Exit(1)
    return ctx


def _generate_id() -> str:
    """Generate a short, stable ID for a schedule entry."""
    return uuid.uuid4().hex[:8]


def _truncate(text: str, max_len: int = 50) -> str:
    """Truncate text with ellipsis if it exceeds max length."""
    return f"{text[:max_len]}..." if len(text) > max_len else text


def _lookup_chat_title(provider: str, chat_id: str) -> str | None:
    """Look up chat title from chat state file."""
    if not provider or not chat_id:
        return None
    state_path = Path(f"/chats/{provider}/{chat_id}/state.json")
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text())
        return data.get("chat", {}).get("title")
    except (json.JSONDecodeError, OSError):
        return None


def _parse_time(time_str: str, timezone: str) -> datetime | None:
    """Parse time string to UTC datetime.

    Accepts ISO 8601 or natural language ('11pm', 'in 2 hours').

    Args:
        time_str: Time string to parse.
        timezone: User's IANA timezone for interpreting local times.

    Returns:
        UTC datetime if parsing succeeds, None otherwise.
    """
    # Fast path: ISO 8601
    try:
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except ValueError:
        pass

    # Natural language fallback
    import dateparser

    settings: dict = {
        "TIMEZONE": timezone,
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "future",
    }
    parsed = dateparser.parse(time_str, settings=settings)
    if parsed:
        return parsed.astimezone(UTC)
    return None


def _read_entries() -> list[dict]:
    """Read all entries from schedule file."""
    if not SCHEDULE_FILE.exists():
        return []

    entries = []
    with SCHEDULE_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                continue
    return entries


def _write_entries(entries: list[dict]) -> None:
    """Write entries back to schedule file."""
    SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SCHEDULE_FILE.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


@app.command()
def create(
    message: Annotated[str, typer.Argument(help="The task message/prompt to execute")],
    at: Annotated[
        str | None,
        typer.Option(
            "--at",
            help="When to execute (e.g., '7:45am', 'tomorrow at 9am', 'in 2 hours')",
        ),
    ] = None,
    cron: Annotated[
        str | None,
        typer.Option(
            "--cron",
            help="Cron in local time (e.g., '0 8 * * *' for 8am daily, '45 7 * * 1-5' for 7:45am weekdays)",
        ),
    ] = None,
) -> None:
    """Create a scheduled task.

    Examples:
        ash-sb schedule create "Remind me to check the build" --at "tomorrow at 9am"
        ash-sb schedule create "Daily status check" --cron "0 8 * * *"
    """
    # Require routing context
    ctx = _require_routing_context()

    # Validate trigger
    if not at and not cron:
        typer.echo(
            "Error: Must specify either --at (one-time) or --cron (recurring)", err=True
        )
        raise typer.Exit(1)

    if at and cron:
        typer.echo("Error: Cannot specify both --at and --cron. Choose one.", err=True)
        raise typer.Exit(1)

    # Parse and validate --at time (supports ISO 8601 and natural language)
    trigger_time: datetime | None = None
    if at:
        trigger_time = _parse_time(at, ctx["timezone"])
        if trigger_time is None:
            typer.echo(f"Error: Could not parse time: {at}", err=True)
            raise typer.Exit(1)
        if trigger_time <= datetime.now(UTC):
            typer.echo(f"Error: --at must be in the future. Got: {at}", err=True)
            raise typer.Exit(1)

    # Validate cron format
    if cron:
        try:
            from croniter import croniter

            croniter(cron)
        except ImportError:
            # croniter not available in sandbox - accept the cron and let server validate
            pass
        except Exception as e:
            typer.echo(f"Error: Invalid cron expression: {e}", err=True)
            raise typer.Exit(1) from None

    # Build entry with stable ID
    entry_id = _generate_id()
    entry: dict = {
        "id": entry_id,
        "message": message,
    }

    if trigger_time:
        entry["trigger_at"] = trigger_time.isoformat().replace("+00:00", "Z")
    if cron:
        entry["cron"] = cron

    # Add routing context (chat_id and provider guaranteed by _require_routing_context)
    entry["chat_id"] = ctx["chat_id"]
    entry["provider"] = ctx["provider"]
    if ctx["chat_title"]:
        entry["chat_title"] = ctx["chat_title"]
    if ctx["user_id"]:
        entry["user_id"] = ctx["user_id"]
    if ctx["username"]:
        entry["username"] = ctx["username"]

    entry["created_at"] = datetime.now(UTC).isoformat()

    # Append to schedule file
    SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with SCHEDULE_FILE.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        typer.echo(f"Error: Failed to write schedule: {e}", err=True)
        raise typer.Exit(1) from None

    # Confirmation with ID
    preview = _truncate(message)
    if trigger_time:
        local_time = _format_time_local(entry["trigger_at"], ctx["timezone"])
        typer.echo(f"Scheduled reminder (id={entry_id})")
        typer.echo(f"  Time: {local_time} ({ctx['timezone']})")
        typer.echo(f"  UTC:  {entry['trigger_at']}")
        typer.echo(f"  Task: {preview}")
    else:
        typer.echo(f"Scheduled recurring task (id={entry_id}) ({cron}): {preview}")


def _filter_by_user(entries: list[dict]) -> list[dict]:
    """Filter entries to only those owned by the current user."""
    user_id = os.environ.get("ASH_USER_ID")
    if not user_id:
        return entries  # No user context, show all
    return [e for e in entries if e.get("user_id") == user_id]


def _format_time_local(iso_time: str, timezone: str) -> str:
    """Format an ISO timestamp in the user's local timezone.

    Args:
        iso_time: ISO 8601 timestamp string.
        timezone: IANA timezone name.

    Returns:
        Formatted local time string.
    """
    from zoneinfo import ZoneInfo

    try:
        # Parse the ISO timestamp
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        # Convert to user's timezone
        tz = ZoneInfo(timezone)
        local_dt = dt.astimezone(tz)
        # Format for display (without timezone info, since we show it in header)
        return local_dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        # Fall back to original format if parsing fails
        return iso_time[:16]


@app.command("list")
def list_tasks() -> None:
    """List scheduled tasks for the current user."""
    ctx = _get_context()
    timezone = ctx["timezone"]
    entries = _filter_by_user(_read_entries())

    if not entries:
        typer.echo("No scheduled tasks found.")
        return

    # Show timezone in header
    typer.echo(f"Scheduled tasks (times shown in {timezone}):\n")

    # Simple table output with target and scheduled_by columns
    typer.echo(
        f"{'ID':<10} {'Type':<10} {'Target':<18} {'By':<12} {'Schedule':<18} {'Message'}"
    )
    typer.echo("-" * 110)

    for entry in entries:
        entry_id = entry.get("id", "?")
        task_type = "periodic" if "cron" in entry else "one-shot"

        # Show provider:title (fallback to provider:chat_id[:10])
        chat_title = entry.get("chat_title") or _lookup_chat_title(
            entry.get("provider", ""), entry.get("chat_id", "")
        )
        if chat_title:
            target = f"{entry.get('provider', '?')}:{chat_title}"
        else:
            chat_id = entry.get("chat_id", "?")
            truncated_chat = chat_id[:10] if len(chat_id) > 10 else chat_id
            target = f"{entry.get('provider', '?')}:{truncated_chat}"

        # Show who scheduled it
        username = entry.get("username", "")
        scheduled_by = f"@{username}" if username else "?"

        message_preview = _truncate(entry.get("message", ""), max_len=25)

        if "cron" in entry:
            schedule = entry["cron"]
        elif "trigger_at" in entry:
            schedule = _format_time_local(entry["trigger_at"], timezone)
        else:
            schedule = "?"

        typer.echo(
            f"{entry_id:<10} {task_type:<10} {target:<18} {scheduled_by:<12} "
            f"{schedule:<18} {message_preview}"
        )

    typer.echo(f"\nTotal: {len(entries)} task(s)")


@app.command()
def cancel(
    entry_id: Annotated[
        str, typer.Option("--id", "-i", help="Entry ID to cancel (8-char hex)")
    ],
) -> None:
    """Cancel a scheduled task by ID (must be owned by current user)."""
    user_id = os.environ.get("ASH_USER_ID")
    entries = _read_entries()

    # Find entry
    found = None
    remaining = []
    for entry in entries:
        if entry.get("id") == entry_id:
            found = entry
        else:
            remaining.append(entry)

    if not found:
        typer.echo(f"Error: No task found with ID {entry_id}", err=True)
        raise typer.Exit(1)

    # Check ownership if user context is available
    if user_id and found.get("user_id") != user_id:
        typer.echo(f"Error: Task {entry_id} does not belong to you", err=True)
        raise typer.Exit(1)

    # Rewrite file without the cancelled entry
    _write_entries(remaining)

    typer.echo(f"Cancelled: {_truncate(found.get('message', ''))}")
