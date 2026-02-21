"""Shared UX helpers for doctor-style commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ash.cli.console import console


@dataclass
class DoctorFinding:
    """A single doctor check result."""

    level: Literal["ok", "warning", "error"]
    check: str
    detail: str
    repair: str | None = None


@dataclass
class DoctorResult:
    """Aggregated doctor output with common summary semantics."""

    findings: list[DoctorFinding]

    @property
    def checks(self) -> int:
        return len(self.findings)

    @property
    def ok_count(self) -> int:
        return sum(1 for finding in self.findings if finding.level == "ok")

    @property
    def warning_count(self) -> int:
        return sum(1 for finding in self.findings if finding.level == "warning")

    @property
    def error_count(self) -> int:
        return sum(1 for finding in self.findings if finding.level == "error")

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def summary_text(self) -> str:
        return (
            f"checks={self.checks} ok={self.ok_count} "
            f"warnings={self.warning_count} errors={self.error_count}"
        )


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
