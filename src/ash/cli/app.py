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
    host: Annotated[
        str,
        typer.Option(
            "--host",
            "-h",
            help="Host to bind to",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Port to bind to",
        ),
    ] = 8080,
) -> None:
    """Start the Ash assistant server."""
    import asyncio

    from rich.console import Console

    console = Console()

    async def run_server() -> None:
        import uvicorn

        from ash.config import WorkspaceLoader, load_config
        from ash.core import Agent, AgentConfig
        from ash.db import init_database
        from ash.llm import create_registry
        from ash.providers.telegram import TelegramProvider
        from ash.server.app import create_app
        from ash.tools import BashTool, ToolExecutor, ToolRegistry, WebSearchTool

        # Load configuration
        console.print("[bold]Loading configuration...[/bold]")
        ash_config = load_config(config)

        # Initialize database
        console.print("[bold]Initializing database...[/bold]")
        database = init_database(database_path=ash_config.memory.database_path)
        await database.connect()

        # Load workspace
        console.print("[bold]Loading workspace...[/bold]")
        workspace_loader = WorkspaceLoader(ash_config.workspace)
        workspace_loader.ensure_workspace()
        workspace = workspace_loader.load()

        # Set up LLM
        console.print("[bold]Setting up LLM providers...[/bold]")
        llm_registry = create_registry()
        llm = llm_registry.get(ash_config.default_llm.provider)

        # Set up tools
        console.print("[bold]Setting up tools...[/bold]")
        tool_registry = ToolRegistry()
        tool_registry.register(BashTool())
        if ash_config.brave_search and ash_config.brave_search.api_key:
            tool_registry.register(
                WebSearchTool(api_key=ash_config.brave_search.api_key)
            )
        tool_executor = ToolExecutor(tool_registry)

        # Create agent
        agent = Agent(
            llm=llm,
            tool_executor=tool_executor,
            workspace=workspace,
            config=AgentConfig(
                model=ash_config.default_llm.model,
                max_tokens=ash_config.default_llm.max_tokens,
                temperature=ash_config.default_llm.temperature,
            ),
        )

        # Set up Telegram if configured
        telegram_provider = None
        if ash_config.telegram and ash_config.telegram.bot_token:
            console.print("[bold]Setting up Telegram provider...[/bold]")
            webhook_url = ash_config.telegram.webhook_url if webhook else None
            telegram_provider = TelegramProvider(
                bot_token=ash_config.telegram.bot_token,
                allowed_users=ash_config.telegram.allowed_users,
                webhook_url=webhook_url,
            )

        # Create FastAPI app
        console.print("[bold]Creating server...[/bold]")
        fastapi_app = create_app(
            database=database,
            agent=agent,
            telegram_provider=telegram_provider,
        )

        # Start server
        console.print(
            f"[bold green]Server starting on http://{host}:{port}[/bold green]"
        )

        if telegram_provider and not webhook:
            # Run both uvicorn and telegram polling
            console.print("[bold]Starting Telegram polling...[/bold]")

            async def start_telegram():
                handler = await fastapi_app.state.server.get_telegram_handler()
                if handler:
                    await telegram_provider.start(handler.handle_message)

            # Start both concurrently
            uvicorn_config = uvicorn.Config(
                fastapi_app, host=host, port=port, log_level="info"
            )
            server = uvicorn.Server(uvicorn_config)

            await asyncio.gather(
                server.serve(),
                start_telegram(),
            )
        else:
            # Just run uvicorn
            uvicorn_config = uvicorn.Config(
                fastapi_app, host=host, port=port, log_level="info"
            )
            server = uvicorn.Server(uvicorn_config)
            await server.serve()

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Server stopped[/bold yellow]")


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
