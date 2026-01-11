"""Server command for running the Ash service."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from ash.cli.console import console


def register(app: typer.Typer) -> None:
    """Register the serve command."""

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
        try:
            asyncio.run(_run_server(config, webhook, host, port))
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Server stopped[/bold yellow]")


async def _run_server(
    config_path: Path | None = None,
    webhook: bool = False,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Run the server asynchronously."""
    import logging
    import signal as signal_module

    import uvicorn

    # Configure logging for all modules with consistent format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        force=True,  # Override any existing configuration
    )

    # Configure uvicorn/aiogram loggers to use same format
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "aiogram"):
        lib_logger = logging.getLogger(logger_name)
        lib_logger.handlers = []  # Remove default handlers
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(log_format))
        lib_logger.addHandler(handler)
        lib_logger.propagate = False

    logger = logging.getLogger(__name__)

    from ash.config import WorkspaceLoader, load_config
    from ash.config.paths import get_pid_path
    from ash.core import create_agent
    from ash.db import init_database
    from ash.providers.telegram import TelegramProvider
    from ash.server.app import create_app
    from ash.service.pid import remove_pid_file, write_pid_file

    # Write PID file for service management
    pid_path = get_pid_path()
    write_pid_file(pid_path)

    # Load configuration
    console.print("[bold]Loading configuration...[/bold]")
    ash_config = load_config(config_path)

    # Initialize Sentry for server mode
    if ash_config.sentry:
        from ash.observability import init_sentry

        if init_sentry(ash_config.sentry, server_mode=True):
            console.print("[dim]Sentry initialized[/dim]")

    # Initialize database
    console.print("[bold]Initializing database...[/bold]")
    database = init_database(database_path=ash_config.memory.database_path)
    await database.connect()

    # Load workspace
    console.print("[bold]Loading workspace...[/bold]")
    workspace_loader = WorkspaceLoader(ash_config.workspace)
    workspace_loader.ensure_workspace()
    workspace = workspace_loader.load()

    # Create agent with all dependencies
    # Create a persistent session for memory tools (remember, recall)
    # This session lives for the duration of the server
    console.print("[bold]Setting up agent...[/bold]")
    memory_session = await database.session().__aenter__()
    components = await create_agent(
        config=ash_config,
        workspace=workspace,
        db_session=memory_session,
        model_alias="default",
    )
    agent = components.agent

    console.print(f"[dim]Tools: {', '.join(components.tool_registry.names)}[/dim]")
    if components.skill_registry:
        console.print(f"[dim]Skills: {len(components.skill_registry)} discovered[/dim]")

    # Set up Telegram if configured
    telegram_provider = None
    if ash_config.telegram and ash_config.telegram.bot_token:
        console.print("[bold]Setting up Telegram provider...[/bold]")
        webhook_url = ash_config.telegram.webhook_url if webhook else None
        telegram_provider = TelegramProvider(
            bot_token=ash_config.telegram.bot_token.get_secret_value(),
            allowed_users=ash_config.telegram.allowed_users,
            webhook_url=webhook_url,
            allowed_groups=ash_config.telegram.allowed_groups,
            group_mode=ash_config.telegram.group_mode,
        )

    # Create FastAPI app
    console.print("[bold]Creating server...[/bold]")
    fastapi_app = create_app(
        database=database,
        agent=agent,
        telegram_provider=telegram_provider,
    )

    # Start server
    console.print(f"[bold green]Server starting on http://{host}:{port}[/bold green]")

    try:
        uvicorn_config = uvicorn.Config(
            fastapi_app,
            host=host,
            port=port,
            log_level="info",
            log_config=None,  # Use our logging config, not uvicorn's
        )
        server = uvicorn.Server(uvicorn_config)

        # Track tasks for cleanup
        telegram_task: asyncio.Task | None = None
        shutdown_event = asyncio.Event()

        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()

        def handle_signal():
            server.should_exit = True
            shutdown_event.set()
            # Cancel telegram polling if running
            if telegram_task and not telegram_task.done():
                telegram_task.cancel()

        for sig in (signal_module.SIGTERM, signal_module.SIGINT):
            loop.add_signal_handler(sig, handle_signal)

        if telegram_provider and not webhook:
            # Run both uvicorn and telegram polling
            console.print("[bold]Starting Telegram polling...[/bold]")

            async def start_telegram():
                # Wait for server to be ready and handler to be created
                handler = None
                for _ in range(50):  # Wait up to 5 seconds
                    handler = await fastapi_app.state.server.get_telegram_handler()
                    if handler:
                        break
                    await asyncio.sleep(0.1)

                if handler:
                    try:
                        await telegram_provider.start(handler.handle_message)
                    except asyncio.CancelledError:
                        logger.info("Telegram polling cancelled")
                else:
                    console.print(
                        "[red]Failed to get Telegram handler after timeout[/red]"
                    )

            telegram_task = asyncio.create_task(start_telegram())
            try:
                await asyncio.gather(server.serve(), telegram_task)
            except asyncio.CancelledError:
                pass
        else:
            await server.serve()
    finally:
        # Stop telegram provider gracefully
        if telegram_provider:
            try:
                await telegram_provider.stop()
            except Exception as e:
                logger.warning(f"Error stopping Telegram provider: {e}")

        # Clean up PID file on exit
        remove_pid_file(pid_path)
