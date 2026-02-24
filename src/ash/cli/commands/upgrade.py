"""Upgrade command for running setup tasks."""

import typer

from ash.cli.console import (
    console,
    dim,
    error,
    info,
    success,
    warning,
)


def register(app: typer.Typer) -> None:
    """Register the upgrade command."""

    @app.command()
    def upgrade() -> None:
        """Upgrade Ash (build sandbox)."""
        console.print("[bold]Upgrading Ash...[/bold]\n")

        _migrate_schedule_into_graph()

        # Build sandbox
        info("Building sandbox...")

        from ash.cli.commands.sandbox import _get_dockerfile_path, _sandbox_build

        dockerfile_path = _get_dockerfile_path()
        if not dockerfile_path:
            error("Dockerfile.sandbox not found")
            dim("Sandbox build skipped")
        elif not _sandbox_build(dockerfile_path):
            warning("Sandbox build failed (retry with 'ash sandbox build')")

        console.print("\n[bold green]Upgrade complete![/bold green]")


def _migrate_schedule_into_graph() -> None:
    """Migrate legacy schedule.jsonl entries into graph schedule nodes."""
    from ash.config.paths import get_ash_home, get_graph_dir
    from ash.scheduling import ScheduleStore
    from ash.scheduling.types import ScheduleEntry

    schedule_file = get_ash_home() / "schedule.jsonl"
    if not schedule_file.exists():
        dim("No legacy schedule file found (skipping schedule migration)")
        return

    graph_schedules = get_graph_dir() / "schedules.jsonl"
    if graph_schedules.exists() and graph_schedules.stat().st_size > 0:
        dim("Graph schedule storage already initialized (skipping legacy migration)")
        return

    info("Migrating schedule entries into ash.graph...")
    try:
        store = ScheduleStore(get_graph_dir())
        imported = 0
        for line_number, line in enumerate(schedule_file.read_text().splitlines()):
            entry = ScheduleEntry.from_line(line, line_number)
            if entry is None:
                continue
            store.add_entry(entry)
            imported += 1

        schedule_file.unlink(missing_ok=True)

        success(
            f"Schedule graph ready ({imported} entr{'y' if imported == 1 else 'ies'} migrated)"
        )
    except Exception as exc:
        warning(f"Schedule migration failed ({exc})")
        dim("Retry by running `ash schedule list` after fixing data issues")
