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
    import subprocess
    import sys

    from rich.console import Console

    console = Console()

    if action == "migrate":
        console.print(f"[bold]Running migrations to {revision}...[/bold]")
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", revision],
            capture_output=False,
        )
        if result.returncode == 0:
            console.print("[green]Migrations completed successfully[/green]")
        else:
            console.print("[red]Migration failed[/red]")
            raise typer.Exit(1)

    elif action == "rollback":
        target = revision if revision != "head" else "-1"
        console.print(f"[bold]Rolling back to {target}...[/bold]")
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "downgrade", target],
            capture_output=False,
        )
        if result.returncode == 0:
            console.print("[green]Rollback completed successfully[/green]")
        else:
            console.print("[red]Rollback failed[/red]")
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
