"""Init command for creating configuration files."""

from pathlib import Path
from typing import Annotated

import typer

from ash.cli.console import console, error, success


def register(app: typer.Typer) -> None:
    """Register the init command."""

    @app.command()
    def init(
        path: Annotated[
            Path | None,
            typer.Option(
                "--path",
                "-p",
                help="Path to config file (default: ~/.ash/config.toml)",
            ),
        ] = None,
    ) -> None:
        """Initialize a new Ash configuration file with sensible defaults."""
        from ash.cli.context import generate_config_template
        from ash.config.paths import get_config_path

        config_path = path.expanduser() if path else get_config_path()

        if config_path.exists():
            error(f"Config file already exists at {config_path}")
            console.print("Use --path to specify a different location")
            raise typer.Exit(1)

        # Create parent directory and write config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(generate_config_template())
        success(f"Created config file at {config_path}")
        console.print("Add your API key, then run: [cyan]ash upgrade[/cyan]")
