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
        "provider": os.environ.get("ASH_PROVIDER", ""),
        "username": os.environ.get("ASH_USERNAME", ""),
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
            help="ISO 8601 UTC timestamp for one-time execution (e.g., 2026-01-12T09:00:00Z)",
        ),
    ] = None,
    cron: Annotated[
        str | None,
        typer.Option(
            "--cron",
            help="Cron expression for recurring execution (e.g., '0 8 * * *' for daily 8am)",
        ),
    ] = None,
) -> None:
    """Create a scheduled task.

    Examples:
        ash-sb schedule create "Remind me to check the build" --at 2026-01-12T10:00:00Z
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

    # Validate --at format and ensure it's in the future
    if at:
        try:
            trigger_time = datetime.fromisoformat(at.replace("Z", "+00:00"))
            if trigger_time <= datetime.now(UTC):
                typer.echo(f"Error: --at must be in the future. Got: {at}", err=True)
                raise typer.Exit(1)
        except ValueError as e:
            typer.echo(f"Error: Invalid --at format: {e}", err=True)
            raise typer.Exit(1) from None

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

    if at:
        entry["trigger_at"] = at
    if cron:
        entry["cron"] = cron

    # Add routing context
    if ctx["chat_id"]:
        entry["chat_id"] = ctx["chat_id"]
    if ctx["user_id"]:
        entry["user_id"] = ctx["user_id"]
    if ctx["username"]:
        entry["username"] = ctx["username"]
    if ctx["provider"]:
        entry["provider"] = ctx["provider"]

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
    preview = f"{message[:50]}..." if len(message) > 50 else message
    if at:
        typer.echo(f"Scheduled one-time task (id={entry_id}) for {at}: {preview}")
    else:
        typer.echo(f"Scheduled recurring task (id={entry_id}) ({cron}): {preview}")


def _filter_by_user(entries: list[dict]) -> list[dict]:
    """Filter entries to only those owned by the current user."""
    user_id = os.environ.get("ASH_USER_ID")
    if not user_id:
        return entries  # No user context, show all
    return [e for e in entries if e.get("user_id") == user_id]


@app.command("list")
def list_tasks() -> None:
    """List scheduled tasks for the current user."""
    entries = _filter_by_user(_read_entries())

    if not entries:
        typer.echo("No scheduled tasks found.")
        return

    # Simple table output
    typer.echo(f"{'ID':<10} {'Type':<10} {'Schedule':<25} {'Message'}")
    typer.echo("-" * 85)

    for entry in entries:
        entry_id = entry.get("id", "?")
        task_type = "periodic" if "cron" in entry else "one-shot"
        message = entry.get("message", "")
        message_preview = f"{message[:35]}..." if len(message) > 35 else message

        if "cron" in entry:
            schedule = entry["cron"]
        elif "trigger_at" in entry:
            schedule = entry["trigger_at"][:19]
        else:
            schedule = "?"

        typer.echo(f"{entry_id:<10} {task_type:<10} {schedule:<25} {message_preview}")

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

    message = found.get("message", "")
    preview = f"{message[:50]}..." if len(message) > 50 else message
    typer.echo(f"Cancelled: {preview}")
