"""Upgrade command for running migrations and setup tasks."""

import subprocess
from pathlib import Path

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
        """Upgrade Ash (run database migrations, build sandbox)."""
        console.print("[bold]Upgrading Ash...[/bold]\n")

        # Ensure data directory exists (for SQLite database)
        data_dir = Path.cwd() / "data"
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            dim(f"Created data directory: {data_dir}")

        # Run database migrations
        info("Running database migrations...")
        try:
            result = subprocess.run(
                ["uv", "run", "alembic", "upgrade", "head"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                if "Running upgrade" in result.stdout or result.stdout.strip():
                    dim(result.stdout.strip())
                success("Database migrations complete")
            else:
                # Check for common issues
                stderr = result.stderr.strip()
                if "Can't locate revision" in stderr:
                    warning("No migrations to run (database is up to date)")
                elif "unable to open database file" in stderr:
                    error("Failed to open database file")
                    dim("Check that data directory exists and is writable")
                else:
                    error("Migration failed")
                    if stderr:
                        # Show just the last meaningful line
                        lines = [
                            line
                            for line in stderr.split("\n")
                            if line.strip() and not line.startswith("  ")
                        ]
                        if lines:
                            dim(lines[-1])
        except FileNotFoundError:
            warning("Alembic not available, skipping migrations")

        # Build sandbox
        console.print()
        info("Building sandbox...")

        from ash.cli.commands.sandbox import _get_dockerfile_path, _sandbox_build

        dockerfile_path = _get_dockerfile_path()
        if not dockerfile_path:
            error("Dockerfile.sandbox not found")
            dim("Sandbox build skipped")
        elif not _sandbox_build(dockerfile_path):
            warning("Sandbox build failed (retry with 'ash sandbox build')")

        console.print("\n[bold green]Upgrade complete![/bold green]")
