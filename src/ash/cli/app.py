"""Main CLI application."""

import typer

from ash.cli.commands import (
    auth,
    browser,
    chat,
    config,
    doctor,
    graph,
    init,
    logs,
    memory,
    people,
    sandbox,
    schedule,
    serve,
    service,
    sessions,
    skill,
    stats,
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
auth.register(app)
init.register(app)
doctor.register(app)
browser.register(app)
serve.register(app)
chat.register(app)
config.register(app)
graph.register(app)
logs.register(app)
memory.register(app)
people.register(app)
schedule.register(app)
sessions.register(app)
upgrade.register(app)
sandbox.register(app)
service.register(app)
skill.register(app)
stats.register(app)

if __name__ == "__main__":
    app()
