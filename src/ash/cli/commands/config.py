"""Configuration management commands."""

from pathlib import Path
from typing import Annotated

import click
import typer

from ash.cli.console import console, error, success


def register(app: typer.Typer) -> None:
    """Register the config command."""

    @app.command()
    def config(
        action: Annotated[
            str | None,
            typer.Argument(help="Action: show, validate"),
        ] = None,
        path: Annotated[
            Path | None,
            typer.Option(
                "--path",
                "-p",
                help="Path to config file (default: $ASH_HOME/config.toml)",
            ),
        ] = None,
    ) -> None:
        """Manage configuration."""
        if action is None:
            ctx = click.get_current_context()
            click.echo(ctx.get_help())
            raise typer.Exit(0)

        from pydantic import ValidationError
        from rich.syntax import Syntax
        from rich.table import Table

        from ash.config import load_config
        from ash.config.paths import get_config_path

        expanded_path = path.expanduser() if path else get_config_path()

        if action == "show":
            if not expanded_path.exists():
                error(f"Config file not found: {expanded_path}")
                console.print("Run 'ash init' to create one")
                raise typer.Exit(1)

            # Display raw TOML with syntax highlighting
            content = expanded_path.read_text()
            syntax = Syntax(content, "toml", theme="monokai", line_numbers=True)
            console.print(f"[bold]Config file: {expanded_path}[/bold]\n")
            console.print(syntax)

        elif action == "validate":
            if not expanded_path.exists():
                error(f"Config file not found: {expanded_path}")
                raise typer.Exit(1)

            try:
                config_obj = load_config(expanded_path)

                # Show validation success with summary
                table = Table(title="Configuration Summary")
                table.add_column("Setting", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Workspace", str(config_obj.workspace))

                # Show models
                model_aliases = config_obj.list_models()
                for alias in model_aliases:
                    model = config_obj.get_model(alias)
                    has_key = config_obj.resolve_api_key(alias) is not None
                    key_status = "[green]✓[/green]" if has_key else "[yellow]?[/yellow]"
                    table.add_row(
                        f"Model '{alias}'",
                        f"{model.provider}/{model.model} {key_status}",
                    )

                table.add_row(
                    "Telegram",
                    "configured"
                    if config_obj.telegram and config_obj.telegram.bot_token
                    else "[dim]not configured[/dim]",
                )
                table.add_row(
                    "Brave Search",
                    "configured"
                    if config_obj.brave_search and config_obj.brave_search.api_key
                    else "[dim]not configured[/dim]",
                )
                table.add_row("Database", str(config_obj.memory.database_path))
                table.add_row(
                    "Server", f"{config_obj.server.host}:{config_obj.server.port}"
                )

                success("Configuration is valid!")
                console.print()
                console.print(table)

            except FileNotFoundError as e:
                error(f"File not found: {e}")
                raise typer.Exit(1) from None
            except ValidationError as e:
                error("Configuration validation failed:")
                console.print()
                for err in e.errors():
                    loc = ".".join(str(x) for x in err["loc"])
                    console.print(f"  [yellow]{loc}[/yellow]: {err['msg']}")
                raise typer.Exit(1) from None
            except Exception as e:
                error(f"Error loading config: {e}")
                raise typer.Exit(1) from None

        else:
            error(f"Unknown action: {action}")
            console.print("Valid actions: show, validate")
            raise typer.Exit(1)
