"""Shared UX helpers for doctor-style commands."""

from __future__ import annotations

from ash.cli.console import console


def render_doctor_preview(
    *,
    title: str,
    subcommands: list[str],
    example: str,
) -> None:
    """Render a standard non-mutating doctor preview message."""
    console.print(f"[bold]{title}[/bold] [dim](preview only)[/dim]")
    console.print("No changes were made.")
    console.print("Run one subcommand explicitly with [cyan]--force[/cyan] to apply:")
    console.print(f"- {', '.join(subcommands)}")
    console.print(f"Example: [cyan]{example}[/cyan]")


def render_force_required(command_name: str) -> None:
    """Render a standard force-required message for mutating doctor commands."""
    console.print(
        f"[red]{command_name} subcommands are mutating. Re-run with --force.[/red]"
    )
