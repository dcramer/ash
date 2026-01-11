"""Setup wizard command."""

from typing import Annotated

import typer

from ash.cli.console import console
from ash.cli.context import generate_config_template


def register(app: typer.Typer) -> None:
    """Register the setup command."""

    @app.command()
    def setup(
        section: Annotated[
            str | None,
            typer.Option(
                "--section",
                "-s",
                help="Configure specific section only (models, telegram, search, advanced)",
            ),
        ] = None,
        reconfigure: Annotated[
            bool,
            typer.Option(
                "--reconfigure",
                "-r",
                help="Reconfigure existing config file",
            ),
        ] = False,
    ) -> None:
        """Interactive setup wizard for Ash configuration.

        Guides you through configuring:
        - LLM provider and model selection
        - Telegram bot integration (optional)
        - Web search with Brave API (optional)
        - Advanced settings like sandbox and server (optional)

        Examples:
            ash setup                    # Full interactive setup
            ash setup --section models   # Configure only models
            ash setup --reconfigure      # Reconfigure existing config
        """
        from rich.prompt import Confirm

        from ash.cli.setup import SetupWizard
        from ash.config.paths import get_config_path

        config_path = get_config_path()

        # If no config exists, generate template first
        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(generate_config_template())
            console.print(f"[green]Created config template at {config_path}[/green]")
            console.print()
        elif not reconfigure:
            console.print(f"[yellow]Config file already exists:[/yellow] {config_path}")
            if not Confirm.ask("Reconfigure?", default=False):
                console.print("[dim]Use --reconfigure to force reconfiguration.[/dim]")
                raise typer.Exit(0)

        wizard = SetupWizard(config_path=config_path)
        sections = [section] if section else None

        if wizard.run(sections=sections):
            raise typer.Exit(0)
        else:
            raise typer.Exit(1)
