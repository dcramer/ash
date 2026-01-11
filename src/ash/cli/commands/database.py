"""Database management commands."""

import subprocess
import sys
from typing import Annotated

import typer

from ash.cli.console import console, error, success


def register(app: typer.Typer) -> None:
    """Register the db command."""

    @app.command("db")
    def db(
        action: Annotated[
            str,
            typer.Argument(help="Action: migrate, rollback, status"),
        ],
        revision: Annotated[
            str,
            typer.Option(
                "--revision",
                "-r",
                help="Target revision (for migrate/rollback)",
            ),
        ] = "head",
    ) -> None:
        """Manage database migrations."""
        if action == "migrate":
            console.print(f"[bold]Running migrations to {revision}...[/bold]")
            result = subprocess.run(
                [sys.executable, "-m", "alembic", "upgrade", revision],
                capture_output=False,
            )
            if result.returncode == 0:
                success("Migrations completed successfully")
            else:
                error("Migration failed")
                raise typer.Exit(1)

        elif action == "rollback":
            target = revision if revision != "head" else "-1"
            console.print(f"[bold]Rolling back to {target}...[/bold]")
            result = subprocess.run(
                [sys.executable, "-m", "alembic", "downgrade", target],
                capture_output=False,
            )
            if result.returncode == 0:
                success("Rollback completed successfully")
            else:
                error("Rollback failed")
                raise typer.Exit(1)

        elif action == "status":
            console.print("[bold]Migration status:[/bold]")
            subprocess.run(
                [sys.executable, "-m", "alembic", "current"],
                capture_output=False,
            )
            console.print("\n[bold]Pending migrations:[/bold]")
            subprocess.run(
                [sys.executable, "-m", "alembic", "history", "--indicate-current"],
                capture_output=False,
            )

        else:
            error(f"Unknown action: {action}")
            raise typer.Exit(1)
