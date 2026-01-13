"""Schedule management commands."""

from typing import Annotated

import typer

from ash.cli.console import console, dim, error, success, warning


def register(app: typer.Typer) -> None:
    """Register the schedule command."""

    @app.command()
    def schedule(
        action: Annotated[
            str,
            typer.Argument(help="Action: list, cancel, clear"),
        ],
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

        Scheduled tasks are stored in workspace/schedule.jsonl.

        Examples:
            ash schedule list                  # List all scheduled tasks
            ash schedule cancel --id a1b2c3d4  # Cancel task by ID
            ash schedule clear                 # Clear all scheduled tasks
        """
        from ash.config import load_config

        config = load_config()
        schedule_file = config.workspace / "schedule.jsonl"

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

    from ash.events.schedule import ScheduleWatcher

    watcher = ScheduleWatcher(schedule_file)
    entries = watcher.get_entries()

    if not entries:
        warning("No scheduled tasks found")
        return

    table = Table(show_header=True)
    table.add_column("ID", style="dim")
    table.add_column("Type")
    table.add_column("Message")
    table.add_column("Schedule")
    table.add_column("Provider")
    table.add_column("Due")

    for entry in entries:
        entry_type = "periodic" if entry.is_periodic else "one-shot"
        message = (
            entry.message[:40] + "..." if len(entry.message) > 40 else entry.message
        )
        due = "[green]yes[/green]" if entry.is_due() else "[dim]no[/dim]"

        # Determine schedule display
        if entry.is_periodic:
            schedule = entry.cron
        elif entry.trigger_at:
            schedule = str(entry.trigger_at)[:19]
        else:
            schedule = "?"

        table.add_row(
            entry.id or "[dim]?[/dim]",
            entry_type,
            message,
            schedule,
            entry.provider or "[dim]none[/dim]",
            due,
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
