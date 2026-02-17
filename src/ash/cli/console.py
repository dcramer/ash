"""Shared console utilities for CLI commands."""

from __future__ import annotations

import subprocess
from enum import Enum
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from ash.config.models import AshConfig
    from ash.llm.base import LLMProvider


class DockerStatus(Enum):
    """Docker availability status."""

    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    NOT_RUNNING = "not_running"


def check_docker() -> DockerStatus:
    """Check if Docker is installed and running.

    Returns:
        DockerStatus indicating the state of Docker.
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return DockerStatus.AVAILABLE
        return DockerStatus.NOT_RUNNING
    except FileNotFoundError:
        return DockerStatus.NOT_INSTALLED


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


def warn_docker_unavailable(status: DockerStatus) -> None:
    """Print a warning about Docker not being available.

    Args:
        status: The Docker status to warn about.
    """
    if status == DockerStatus.NOT_INSTALLED:
        warning("Docker is not installed")
        dim("Install Docker from https://docs.docker.com/get-docker/")
    elif status == DockerStatus.NOT_RUNNING:
        warning("Docker is not running")
        dim("Start Docker to enable sandbox functionality")


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
    for name, style_or_kwargs in columns:
        if isinstance(style_or_kwargs, dict):
            table.add_column(name, **style_or_kwargs)
        else:
            table.add_column(name, style=style_or_kwargs)
    return table


def confirm_or_cancel(prompt: str, force: bool) -> bool:
    """Return True if the user confirms (or force is set). Print cancel on decline."""
    if force:
        return True
    confirmed = typer.confirm(prompt)
    if not confirmed:
        dim("Cancelled")
    return confirmed


def create_llm(
    config: AshConfig, model_alias: str | None = None
) -> tuple[LLMProvider, str]:
    """Create an LLM provider from config. Returns (provider, model_name)."""
    from ash.llm import create_llm_provider

    alias = model_alias or "default"
    model_config = config.get_model(alias)
    api_key = config.resolve_api_key(alias)
    llm = create_llm_provider(
        model_config.provider,
        api_key=api_key.get_secret_value() if api_key else None,
    )
    return llm, model_config.model
