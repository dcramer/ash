"""Server command for running the Ash service."""

import asyncio
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

logger = logging.getLogger(__name__)


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
            # Use print here since logging may not be configured yet
            print("\nServer stopped")


async def _run_server(
    config_path: Path | None = None,
    webhook: bool = False,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Run the server asynchronously."""
    import signal as signal_module

    import uvicorn

    from ash.logging import configure_logging

    # Configure logging with Rich for colorful server output and file logging
    configure_logging(use_rich=True, log_to_file=True)

    from ash.config import WorkspaceLoader, load_config
    from ash.config.paths import get_pid_path, get_rpc_socket_path
    from ash.core import create_agent
    from ash.db import init_database
    from ash.providers.telegram import TelegramProvider
    from ash.rpc import RPCServer, register_memory_methods
    from ash.server.app import create_app
    from ash.service.pid import remove_pid_file, write_pid_file

    # Write PID file for service management
    pid_path = get_pid_path()
    write_pid_file(pid_path)

    # Load configuration
    logger.info("Loading configuration")
    ash_config = load_config(config_path)

    # Initialize Sentry for server mode
    if ash_config.sentry:
        from ash.observability import init_sentry

        if init_sentry(ash_config.sentry, server_mode=True):
            logger.info("Sentry initialized")

    # Initialize database
    logger.info("Initializing database")
    database = init_database(database_path=ash_config.memory.database_path)
    await database.connect()

    # Load workspace
    logger.info("Loading workspace")
    workspace_loader = WorkspaceLoader(ash_config.workspace)
    workspace_loader.ensure_workspace()
    workspace = workspace_loader.load()

    # Create agent with all dependencies
    # Create a persistent session for memory tools (remember, recall)
    # This session lives for the duration of the server
    # Use the factory directly to avoid the auto-commit context manager
    logger.info("Setting up agent")
    memory_session = database.session_factory()
    components = await create_agent(
        config=ash_config,
        workspace=workspace,
        db_session=memory_session,
        model_alias="default",
    )
    agent = components.agent

    # Run memory garbage collection on startup if enabled
    if ash_config.memory.auto_gc and components.memory_manager:
        logger.debug("Running memory garbage collection")
        expired, superseded = await components.memory_manager.gc()
        if expired or superseded:
            logger.info(
                f"Cleaned up {expired} expired, {superseded} superseded memories"
            )

    # Start RPC server for sandbox communication
    rpc_server: RPCServer | None = None
    if components.memory_manager:
        rpc_socket_path = get_rpc_socket_path()
        rpc_server = RPCServer(rpc_socket_path)
        register_memory_methods(rpc_server, components.memory_manager)
        await rpc_server.start()
        logger.info(f"RPC server started at {rpc_socket_path}")

    logger.debug(f"Tools: {', '.join(components.tool_registry.names)}")
    if components.skill_registry:
        logger.debug(f"Skills: {len(components.skill_registry)} discovered")

    # Set up Telegram if configured
    telegram_provider = None
    if ash_config.telegram and ash_config.telegram.bot_token:
        logger.info("Setting up Telegram provider")
        webhook_url = ash_config.telegram.webhook_url if webhook else None
        telegram_provider = TelegramProvider(
            bot_token=ash_config.telegram.bot_token.get_secret_value(),
            allowed_users=ash_config.telegram.allowed_users,
            webhook_url=webhook_url,
            allowed_groups=ash_config.telegram.allowed_groups,
            group_mode=ash_config.telegram.group_mode,
        )

    # Set up schedule watcher
    from ash.events import ScheduledTaskHandler, ScheduleWatcher

    schedule_file = ash_config.workspace / "schedule.jsonl"
    schedule_watcher = ScheduleWatcher(schedule_file)

    # Build sender map from available providers
    senders: dict[str, Any] = {}
    if telegram_provider:
        senders["telegram"] = telegram_provider.send_message

    # Create and register handler if we have senders
    if senders:
        schedule_handler = ScheduledTaskHandler(agent, senders)
        schedule_watcher.add_handler(schedule_handler.handle)
        logger.debug(f"Schedule watcher: {schedule_file}")
    else:
        logger.debug("Schedule watcher disabled (no providers)")

    # Create FastAPI app
    logger.info("Creating server")
    fastapi_app = create_app(
        database=database,
        agent=agent,
        telegram_provider=telegram_provider,
        config=ash_config,
        agent_registry=components.agent_registry,
        skill_registry=components.skill_registry,
    )

    # Start server
    logger.info(f"Server starting on http://{host}:{port}")

    try:
        uvicorn_config = uvicorn.Config(
            fastapi_app,
            host=host,
            port=port,
            log_level="info",
            log_config=None,  # Use our logging config, not uvicorn's
        )
        server = uvicorn.Server(uvicorn_config)

        # Start schedule watcher
        if senders:
            await schedule_watcher.start()

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
            logger.info("Starting Telegram polling")

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
                    logger.error("Failed to get Telegram handler after timeout")

            telegram_task = asyncio.create_task(start_telegram())
            try:
                await asyncio.gather(server.serve(), telegram_task)
            except asyncio.CancelledError:
                pass
        else:
            await server.serve()
    finally:
        await _cleanup_server(
            schedule_watcher,
            telegram_provider,
            rpc_server,
            components.sandbox_executor,
            memory_session,
            pid_path,
            remove_pid_file,
        )


async def _cleanup_server(
    schedule_watcher,
    telegram_provider,
    rpc_server,
    sandbox_executor,
    memory_session,
    pid_path,
    remove_pid_file,
) -> None:
    """Clean up server resources."""
    for resource, method in [
        (schedule_watcher, "stop"),
        (telegram_provider, "stop"),
        (rpc_server, "stop"),
        (sandbox_executor, "cleanup"),
        (memory_session, "close"),
    ]:
        if resource:
            try:
                await getattr(resource, method)()
            except Exception as e:
                logger.warning(f"Error during {method}: {e}")

    remove_pid_file(pid_path)
