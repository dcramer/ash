"""Shared console utilities for CLI commands."""

from rich.console import Console
from rich.table import Table

# Shared console instance for all CLI commands
console = Console()


def error(msg: str) -> None:
    """Print an error message in red."""
    console.print(f"[red]{msg}[/red]")


def warning(msg: str) -> None:
    """Print a warning message in yellow."""
    console.print(f"[yellow]{msg}[/yellow]")


def success(msg: str) -> None:
    """Print a success message in green."""
    console.print(f"[green]{msg}[/green]")


def info(msg: str) -> None:
    """Print an info message in cyan."""
    console.print(f"[cyan]{msg}[/cyan]")


def dim(msg: str) -> None:
    """Print a dimmed message."""
    console.print(f"[dim]{msg}[/dim]")


def create_table(
    title: str,
    columns: list[tuple[str, str | dict]],
) -> Table:
    """Create a styled table with consistent formatting.

    Args:
        title: Table title.
        columns: List of (name, style) or (name, kwargs_dict) tuples.

    Returns:
        Configured Rich Table.
    """
    table = Table(title=title)
    for col in columns:
        name = col[0]
        if isinstance(col[1], dict):
            table.add_column(name, **col[1])
        else:
            table.add_column(name, style=col[1])
    return table
