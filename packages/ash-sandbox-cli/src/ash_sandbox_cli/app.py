"""Sandboxed CLI application."""

import typer

from ash_sandbox_cli.commands import (
    browser,
    config,
    logs,
    memory,
    schedule,
    skill,
    todo,
)

app = typer.Typer(
    name="ash",
    help="Ash sandboxed CLI for agent self-service.",
    no_args_is_help=True,
)

# Register command groups
app.add_typer(config.app, name="config")
app.add_typer(logs.app, name="logs")
app.add_typer(memory.app, name="memory")
app.add_typer(browser.app, name="browser")
app.add_typer(schedule.app, name="schedule")
app.add_typer(todo.app, name="todo")
app.add_typer(skill.app, name="skill")


if __name__ == "__main__":
    app()
