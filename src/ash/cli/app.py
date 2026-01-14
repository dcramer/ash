"""Main CLI application."""

import typer

from ash.cli.commands import (
    chat,
    config,
    database,
    init,
    memory,
    sandbox,
    schedule,
    serve,
    service,
    sessions,
    skill,
    upgrade,
)

app = typer.Typer(
    name="ash",
    help="Ash - Personal Assistant Agent",
    no_args_is_help=True,
)


@app.command()
def help(ctx: typer.Context) -> None:
    """Show help information."""
    if ctx.parent:
        print(ctx.parent.get_help())


# Register commands from modules
init.register(app)
serve.register(app)
chat.register(app)
config.register(app)
database.register(app)
memory.register(app)
schedule.register(app)
sessions.register(app)
upgrade.register(app)
sandbox.register(app)
service.register(app)
skill.register(app)

if __name__ == "__main__":
    app()
