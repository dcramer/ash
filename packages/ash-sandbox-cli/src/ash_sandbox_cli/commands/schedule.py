"""Schedule management commands for sandboxed CLI."""

import os
from datetime import UTC, datetime
from typing import Annotated

import typer

from ash_sandbox_cli.rpc import RPCError, rpc_call

app = typer.Typer(
    name="schedule",
    help="Manage scheduled tasks.",
    no_args_is_help=True,
)


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


def _truncate(text: str, max_len: int = 50) -> str:
    """Truncate text with ellipsis if it exceeds max length."""
    return f"{text[:max_len]}..." if len(text) > max_len else text


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


def _format_time_local(iso_time: str, timezone: str) -> str:
    """Format an ISO timestamp in the user's local timezone."""
    from zoneinfo import ZoneInfo

    try:
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        tz = ZoneInfo(timezone)
        local_dt = dt.astimezone(tz)
        return local_dt.strftime("%a %Y-%m-%d %H:%M")
    except Exception:
        return iso_time[:16]


def _format_next_cron(cron_expr: str, timezone: str) -> str | None:
    """Return the next fire time for a cron expression in the given timezone."""
    from zoneinfo import ZoneInfo

    try:
        from croniter import croniter
    except ImportError:
        return None

    try:
        tz = ZoneInfo(timezone)
        now_local = datetime.now(UTC).astimezone(tz)
        it = croniter(cron_expr, now_local)
        next_fire = it.get_next(datetime)
        return next_fire.strftime("%a %Y-%m-%d %H:%M")
    except Exception:
        return None


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
    timezone: Annotated[
        str | None,
        typer.Option(
            "--tz",
            help="Timezone for schedule (IANA name, e.g., 'America/New_York'). Overrides default.",
        ),
    ] = None,
) -> None:
    """Create a scheduled task.

    Examples:
        ash-sb schedule create "Remind me to check the build" --at "tomorrow at 9am"
        ash-sb schedule create "Daily status check" --cron "0 8 * * *"
        ash-sb schedule create "Standup" --cron "0 10 * * 1-5" --tz America/New_York
    """
    ctx = _require_routing_context()
    if timezone:
        ctx["timezone"] = timezone

    if not at and not cron:
        typer.echo(
            "Error: Must specify either --at (one-time) or --cron (recurring)", err=True
        )
        raise typer.Exit(1)

    if at and cron:
        typer.echo("Error: Cannot specify both --at and --cron. Choose one.", err=True)
        raise typer.Exit(1)

    # Parse and validate --at time
    trigger_at_iso: str | None = None
    if at:
        trigger_time = _parse_time(at, ctx["timezone"])
        if trigger_time is None:
            typer.echo(f"Error: Could not parse time: {at}", err=True)
            raise typer.Exit(1)
        if trigger_time <= datetime.now(UTC):
            from zoneinfo import ZoneInfo

            tz = ZoneInfo(ctx["timezone"])
            local_str = trigger_time.astimezone(tz).strftime("%Y-%m-%d %H:%M %Z")
            typer.echo(
                f"Error: Time '{at}' parsed as {local_str} which is in the past",
                err=True,
            )
            raise typer.Exit(1)
        trigger_at_iso = trigger_time.isoformat().replace("+00:00", "Z")

    # Validate cron format
    if cron:
        try:
            from croniter import croniter

            croniter(cron)
        except ImportError:
            pass
        except Exception as e:
            typer.echo(f"Error: Invalid cron expression: {e}", err=True)
            raise typer.Exit(1) from None

    # Build RPC params
    params: dict[str, str | None] = {
        "message": message,
        "chat_id": ctx["chat_id"],
        "provider": ctx["provider"],
        "timezone": ctx["timezone"],
    }
    if trigger_at_iso:
        params["trigger_at"] = trigger_at_iso
    if cron:
        params["cron"] = cron
    if ctx["chat_title"]:
        params["chat_title"] = ctx["chat_title"]
    if ctx["user_id"]:
        params["user_id"] = ctx["user_id"]
    if ctx["username"]:
        params["username"] = ctx["username"]

    try:
        result = rpc_call("schedule.create", params)
    except ConnectionError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except RPCError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    entry_id = result.get("id", "?")
    entry = result.get("entry", {})
    preview = _truncate(message)

    tz = ctx["timezone"]
    if trigger_at_iso:
        local_time = _format_time_local(trigger_at_iso, tz)
        typer.echo(f"Scheduled reminder (id={entry_id})")
        typer.echo(f"  Time: {local_time} ({tz})")
        typer.echo(f"  UTC:  {entry.get('trigger_at', trigger_at_iso)}")
        typer.echo(f"  Task: {preview}")
    else:
        assert cron is not None
        next_fire = _format_next_cron(cron, tz)
        typer.echo(f"Scheduled recurring task (id={entry_id})")
        typer.echo(f"  Cron: {cron} ({tz})")
        if next_fire:
            typer.echo(f"  Next: {next_fire}")
        typer.echo(f"  Task: {preview}")
        if tz == "UTC":
            typer.echo("  Hint: Use --tz to set timezone (e.g. --tz America/New_York)")


@app.command("list")
def list_tasks(
    all_rooms: Annotated[
        bool,
        typer.Option(
            "--all",
            "-a",
            help="Show tasks from all rooms (default: current room only)",
        ),
    ] = False,
) -> None:
    """List scheduled tasks for the current user."""
    ctx = _get_context()
    timezone = ctx["timezone"]

    try:
        params: dict[str, str | None] = {}
        if ctx["user_id"]:
            params["user_id"] = ctx["user_id"]
        if not all_rooms and ctx["chat_id"]:
            params["chat_id"] = ctx["chat_id"]
        entries = rpc_call("schedule.list", params)
    except ConnectionError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except RPCError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    if not entries:
        typer.echo("No scheduled tasks found.")
        return

    typer.echo(f"Scheduled tasks (times shown in {timezone}):\n")
    for entry in entries:
        entry_id = entry.get("id", "?")
        task_type = "periodic" if "cron" in entry else "one-shot"
        message_preview = _truncate(entry.get("message", ""), max_len=40)

        if "cron" in entry:
            entry_tz = entry.get("timezone", timezone)
            schedule = f"{entry['cron']} ({entry_tz})"
            next_fire = _format_next_cron(entry["cron"], entry_tz)
        elif "trigger_at" in entry:
            schedule = (
                f"{_format_time_local(entry['trigger_at'], timezone)} ({timezone})"
            )
            next_fire = None
        else:
            schedule = "?"
            next_fire = None

        typer.echo(f"  {entry_id}  {task_type:<10} {schedule}")
        if all_rooms:
            room_label = entry.get("chat_title") or entry.get("chat_id") or "unknown"
            typer.echo(f"           Room: {room_label}")
        if next_fire:
            typer.echo(f"           Next: {next_fire}")
        typer.echo(f"           Task: {message_preview}")
        typer.echo()

    typer.echo(f"Total: {len(entries)} task(s)")


@app.command()
def cancel(
    entry_id: Annotated[
        str, typer.Option("--id", "-i", help="Entry ID to cancel (8-char hex)")
    ],
) -> None:
    """Cancel a scheduled task by ID (must be owned by current user)."""
    ctx = _get_context()

    try:
        params: dict[str, str | None] = {"entry_id": entry_id}
        if ctx["user_id"]:
            params["user_id"] = ctx["user_id"]
        result = rpc_call("schedule.cancel", params)
    except ConnectionError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except RPCError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    if result.get("cancelled"):
        entry = result.get("entry", {})
        preview = _truncate(entry.get("message", ""), max_len=50)
        typer.echo(f"Cancelled task (id={entry_id}): {preview}")
    else:
        typer.echo(f"Error: No task found with ID {entry_id}", err=True)
        raise typer.Exit(1)


@app.command()
def update(
    entry_id: Annotated[
        str, typer.Option("--id", "-i", help="Entry ID to update (8-char hex)")
    ],
    message: Annotated[
        str | None,
        typer.Option("--message", "-m", help="New message/prompt"),
    ] = None,
    at: Annotated[
        str | None,
        typer.Option(
            "--at",
            help="New trigger time (e.g., '7:45am', 'tomorrow at 9am')",
        ),
    ] = None,
    cron: Annotated[
        str | None,
        typer.Option("--cron", help="New cron expression"),
    ] = None,
    timezone: Annotated[
        str | None,
        typer.Option("--tz", help="New timezone (IANA name)"),
    ] = None,
) -> None:
    """Update a scheduled task by ID.

    Examples:
        ash-sb schedule update --id a1b2c3d4 --message "New reminder text"
        ash-sb schedule update --id a1b2c3d4 --at "tomorrow at 10am"
        ash-sb schedule update --id a1b2c3d4 --cron "0 9 * * *"
    """
    ctx = _get_context()

    if message is None and at is None and cron is None and timezone is None:
        typer.echo(
            "Error: At least one of --message, --at, --cron, or --tz required", err=True
        )
        raise typer.Exit(1)

    # Parse --at time
    trigger_at_iso: str | None = None
    if at is not None:
        trigger_time = _parse_time(at, ctx["timezone"])
        if trigger_time is None:
            typer.echo(f"Error: Could not parse time: {at}", err=True)
            raise typer.Exit(1)
        if trigger_time <= datetime.now(UTC):
            from zoneinfo import ZoneInfo

            tz = ZoneInfo(ctx["timezone"])
            local_str = trigger_time.astimezone(tz).strftime("%Y-%m-%d %H:%M %Z")
            typer.echo(
                f"Error: Time '{at}' parsed as {local_str} which is in the past",
                err=True,
            )
            raise typer.Exit(1)
        trigger_at_iso = trigger_time.isoformat().replace("+00:00", "Z")

    # Validate cron
    if cron is not None:
        try:
            from croniter import croniter

            croniter(cron)
        except ImportError:
            pass
        except Exception as e:
            typer.echo(f"Error: Invalid cron expression: {e}", err=True)
            raise typer.Exit(1) from None

    params: dict[str, str | None] = {"entry_id": entry_id}
    if ctx["user_id"]:
        params["user_id"] = ctx["user_id"]
    if message is not None:
        params["message"] = message
    if trigger_at_iso is not None:
        params["trigger_at"] = trigger_at_iso
    if cron is not None:
        params["cron"] = cron
    if timezone is not None:
        params["timezone"] = timezone

    try:
        result = rpc_call("schedule.update", params)
    except ConnectionError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None
    except RPCError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    if result.get("updated"):
        entry = result.get("entry", {})
        preview = _truncate(entry.get("message", ""))
        entry_tz = entry.get("timezone", ctx["timezone"])
        is_periodic = "cron" in entry
        if is_periodic:
            cron_expr = entry.get("cron", "")
            next_fire = _format_next_cron(cron_expr, entry_tz)
            typer.echo(f"Updated recurring task (id={entry_id})")
            typer.echo(f"  Cron: {cron_expr} ({entry_tz})")
            if next_fire:
                typer.echo(f"  Next: {next_fire}")
            typer.echo(f"  Task: {preview}")
        else:
            local_time = _format_time_local(entry["trigger_at"], ctx["timezone"])
            typer.echo(f"Updated reminder (id={entry_id})")
            typer.echo(f"  Time: {local_time} ({ctx['timezone']})")
            typer.echo(f"  UTC:  {entry['trigger_at']}")
            typer.echo(f"  Task: {preview}")
    else:
        typer.echo(f"Error: No task found with ID {entry_id}", err=True)
        raise typer.Exit(1)
