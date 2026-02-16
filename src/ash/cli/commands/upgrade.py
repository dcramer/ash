"""Upgrade command for running setup tasks."""

import typer

from ash.cli.console import (
    console,
    dim,
    error,
    info,
    warning,
)


def register(app: typer.Typer) -> None:
    """Register the upgrade command."""

    @app.command()
    def upgrade() -> None:
        """Upgrade Ash (build sandbox)."""
        console.print("[bold]Upgrading Ash...[/bold]\n")

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
