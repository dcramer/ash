"""Main CLI application."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="ash",
    help="Ash - Personal Assistant Agent",
    no_args_is_help=True,
)


@app.command()
def serve(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration file",
        ),
    ] = None,
    webhook: Annotated[
        bool,
        typer.Option(
            "--webhook",
            help="Use webhook mode instead of polling",
        ),
    ] = False,
) -> None:
    """Start the Ash assistant server."""
    from rich.console import Console

    console = Console()
    console.print("[bold green]Starting Ash server...[/bold green]")

    # TODO: Implement server startup
    console.print(f"Config: {config or 'default'}")
    console.print(f"Webhook mode: {webhook}")
    console.print("[yellow]Server not yet implemented[/yellow]")


@app.command()
def config(
    action: Annotated[
        str,
        typer.Argument(help="Action: init, show, validate"),
    ],
    path: Annotated[
        Path,
        typer.Option(
            "--path",
            "-p",
            help="Path to config file",
        ),
    ] = Path("~/.ash/config.toml"),
) -> None:
    """Manage configuration."""
    from rich.console import Console

    console = Console()

    if action == "init":
        console.print("[yellow]Config init not yet implemented[/yellow]")
    elif action == "show":
        console.print("[yellow]Config show not yet implemented[/yellow]")
    elif action == "validate":
        console.print("[yellow]Config validate not yet implemented[/yellow]")
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(1)


@app.command()
def db(
    action: Annotated[
        str,
        typer.Argument(help="Action: migrate, rollback, status"),
    ],
) -> None:
    """Manage database migrations."""
    from rich.console import Console

    console = Console()

    if action == "migrate":
        console.print("[yellow]DB migrate not yet implemented[/yellow]")
    elif action == "rollback":
        console.print("[yellow]DB rollback not yet implemented[/yellow]")
    elif action == "status":
        console.print("[yellow]DB status not yet implemented[/yellow]")
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(1)


@app.command()
def memory(
    action: Annotated[
        str,
        typer.Argument(help="Action: search, stats, export, clear"),
    ],
    query: Annotated[
        str | None,
        typer.Option(
            "--query",
            "-q",
            help="Search query",
        ),
    ] = None,
) -> None:
    """Manage conversation memory."""
    from rich.console import Console

    console = Console()

    if action == "search":
        if not query:
            console.print("[red]--query is required for search[/red]")
            raise typer.Exit(1)
        console.print("[yellow]Memory search not yet implemented[/yellow]")
    elif action == "stats":
        console.print("[yellow]Memory stats not yet implemented[/yellow]")
    elif action == "export":
        console.print("[yellow]Memory export not yet implemented[/yellow]")
    elif action == "clear":
        console.print("[yellow]Memory clear not yet implemented[/yellow]")
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
