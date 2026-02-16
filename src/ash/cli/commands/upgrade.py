"""Upgrade command for running migrations and setup tasks."""

import asyncio

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
        """Upgrade Ash (run data migrations, build sandbox)."""
        console.print("[bold]Upgrading Ash...[/bold]\n")

        # Run data migrations (filesystem layout + SQLite -> JSONL)
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

    # SQLite -> JSONL migration (for users upgrading from SQLite-backed storage)
    try:
        from ash.cli.context import get_graph_dir
        from ash.store.migration_export import migrate_sqlite_to_jsonl

        graph_dir = get_graph_dir()
        if await migrate_sqlite_to_jsonl(graph_dir):
            success("Migrated SQLite data to JSONL")
        else:
            dim("Graph data up to date")
    except Exception:
        logger.warning("SQLite to JSONL migration failed", exc_info=True)
        warning("SQLite to JSONL migration failed (see logs)")
