"""Sandboxed CLI application."""

import typer

from ash.sandbox.cli.commands import schedule

app = typer.Typer(
    name="ash",
    help="Ash sandboxed CLI for agent self-service.",
    no_args_is_help=True,
)

# Register command groups
app.add_typer(schedule.app, name="schedule")


if __name__ == "__main__":
    app()
