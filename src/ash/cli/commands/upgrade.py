"""Upgrade command for running migrations and setup tasks."""

import asyncio
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

        # Run Alembic schema migrations
        info("Running schema migrations...")
        try:
            result = subprocess.run(
                ["uv", "run", "alembic", "upgrade", "head"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                if "Running upgrade" in result.stdout or result.stdout.strip():
                    dim(result.stdout.strip())
                success("Schema migrations complete")
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
            warning("Alembic not available, skipping schema migrations")

        # Run data migrations (filesystem layout + JSONL -> SQLite)
        console.print()
        info("Running data migrations...")
        asyncio.run(_run_data_migrations())

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


async def _run_data_migrations() -> None:
    """Run filesystem and data migrations."""
    import logging

    logger = logging.getLogger(__name__)

    # Filesystem layout migration (move files to new locations)
    try:
        from ash.store.migration import migrate_filesystem

        if migrate_filesystem():
            success("Filesystem layout migrated")
        else:
            dim("Filesystem layout up to date")
    except Exception:
        logger.warning("Filesystem migration failed", exc_info=True)
        warning("Filesystem migration failed (see logs)")

    # JSONL directory migration (old scattered paths -> graph/)
    try:
        from ash.memory.migration import migrate_to_graph_dir

        if await migrate_to_graph_dir():
            success("Migrated to graph directory layout")
        else:
            dim("Graph directory layout up to date")
    except Exception:
        logger.warning("Graph directory migration failed", exc_info=True)
        warning("Graph directory migration failed (see logs)")

    # JSONL -> SQLite migration
    try:
        from ash.cli.context import get_config
        from ash.db import init_database

        config = get_config()
        db = init_database(database_path=config.memory.database_path)
        await db.connect()
        try:
            from ash.store.migration_sqlite import migrate_jsonl_to_sqlite

            if await migrate_jsonl_to_sqlite(db):
                success("Migrated JSONL data to SQLite")
            else:
                dim("SQLite data up to date")
        finally:
            await db.disconnect()
    except Exception:
        logger.warning("JSONL to SQLite migration failed", exc_info=True)
        warning("JSONL to SQLite migration failed (see logs)")
