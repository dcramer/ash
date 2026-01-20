"""Schedule management commands."""

from datetime import UTC, datetime
from typing import Annotated

import click
import typer

from ash.cli.console import console, dim, error, success, warning


def _format_countdown(next_fire: datetime | None) -> str:
    """Format a countdown string for the next fire time."""
    if next_fire is None:
        return "[dim]?[/dim]"

    now = datetime.now(UTC)
    if next_fire <= now:
        return "[green]now[/green]"

    delta = next_fire - now
    total_seconds = int(delta.total_seconds())

    if total_seconds < 60:
        return f"in {total_seconds}s"

    total_minutes = total_seconds // 60
    if total_minutes < 60:
        return f"in {total_minutes}m"

    hours = total_minutes // 60
    minutes = total_minutes % 60
    if hours < 24:
        if minutes:
            return f"in {hours}h {minutes}m"
        return f"in {hours}h"

    days = hours // 24
    hours = hours % 24
    if hours:
        return f"in {days}d {hours}h"
    return f"in {days}d"


def register(app: typer.Typer) -> None:
    """Register the schedule command."""

    @app.command()
    def schedule(
        action: Annotated[
            str | None,
            typer.Argument(help="Action: list, cancel, clear"),
        ] = None,
        entry_id: Annotated[
            str | None,
            typer.Option(
                "--id",
                "-i",
                help="Entry ID (8-char hex) for cancel",
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
        """Manage scheduled tasks.

        Scheduled tasks are stored in ~/.ash/schedule.jsonl.

        Examples:
            ash schedule list                  # List all scheduled tasks
            ash schedule cancel --id a1b2c3d4  # Cancel task by ID
            ash schedule clear                 # Clear all scheduled tasks
        """
        if action is None:
            ctx = click.get_current_context()
            click.echo(ctx.get_help())
            raise typer.Exit(0)

        from ash.config.paths import get_schedule_file

        schedule_file = get_schedule_file()

        if action == "list":
            _schedule_list(schedule_file)

        elif action == "cancel":
            if entry_id is None:
                error("--id is required for cancel")
                raise typer.Exit(1)
            _schedule_cancel(schedule_file, entry_id)

        elif action == "clear":
            _schedule_clear(schedule_file, force)

        else:
            error(f"Unknown action: {action}")
            console.print("Valid actions: list, cancel, clear")
            raise typer.Exit(1)


def _schedule_list(schedule_file) -> None:
    """List all scheduled tasks."""
    from rich.table import Table

    from ash.config import load_config
    from ash.events.schedule import ScheduleWatcher

    config = load_config()
    watcher = ScheduleWatcher(schedule_file, timezone=config.timezone)
    entries = watcher.get_entries()

    if not entries:
        warning("No scheduled tasks found")
        return

    table = Table(show_header=True)
    table.add_column("ID", style="dim")
    table.add_column("Type")
    table.add_column("Chat")
    table.add_column("Message")
    table.add_column("Schedule")
    table.add_column("Next Fire")

    for entry in entries:
        entry_type = "periodic" if entry.is_periodic else "one-shot"
        message = (
            entry.message[:40] + "..." if len(entry.message) > 40 else entry.message
        )

        # Display chat_title or truncated chat_id
        if entry.chat_title:
            chat = entry.chat_title
        elif entry.chat_id:
            chat = (
                entry.chat_id[:10] + "..." if len(entry.chat_id) > 10 else entry.chat_id
            )
        else:
            chat = "[dim]none[/dim]"

        # Determine schedule display
        if entry.is_periodic:
            schedule = entry.cron
        elif entry.trigger_at:
            schedule = str(entry.trigger_at)[:19]
        else:
            schedule = "?"

        # Calculate next fire countdown
        next_fire = entry.next_fire_time(config.timezone)
        next_fire_display = _format_countdown(next_fire)

        table.add_row(
            entry.id or "[dim]?[/dim]",
            entry_type,
            chat,
            message,
            schedule,
            next_fire_display,
        )

    console.print(table)
    console.print(f"\n{dim(f'Total: {len(entries)} task(s)')}")


def _schedule_cancel(schedule_file, entry_id: str) -> None:
    """Cancel a scheduled task by ID."""
    from ash.events.schedule import ScheduleWatcher

    watcher = ScheduleWatcher(schedule_file)
    entries = watcher.get_entries()

    # Find the entry to show what we're removing
    entry = next((e for e in entries if e.id == entry_id), None)
    if not entry:
        error(f"No task found with ID {entry_id}")
        raise typer.Exit(1)

    if watcher.remove_entry(entry_id):
        success(f"Cancelled: {entry.message[:50]}...")
    else:
        error(f"Failed to cancel task {entry_id}")
        raise typer.Exit(1)


def _schedule_clear(schedule_file, force: bool) -> None:
    """Clear all scheduled tasks."""
    from ash.events.schedule import ScheduleWatcher

    watcher = ScheduleWatcher(schedule_file)
    stats = watcher.get_stats()

    if stats["total"] == 0:
        warning("No scheduled tasks to clear")
        return

    if not force:
        confirm = typer.confirm(
            f"This will delete {stats['total']} scheduled task(s). Continue?"
        )
        if not confirm:
            console.print("[dim]Cancelled[/dim]")
            return

    count = watcher.clear_all()
    success(f"Cleared {count} scheduled task(s)")
