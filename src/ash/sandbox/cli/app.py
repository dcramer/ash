"""Sandboxed CLI application."""

import typer

from ash.sandbox.cli.commands import memory, schedule, skill

app = typer.Typer(
    name="ash",
    help="Ash sandboxed CLI for agent self-service.",
    no_args_is_help=True,
)

# Register command groups
app.add_typer(memory.app, name="memory")
app.add_typer(schedule.app, name="schedule")
app.add_typer(skill.app, name="skill")


if __name__ == "__main__":
    app()
