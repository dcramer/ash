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
    from ash.config.paths import get_logs_path, get_pid_path, get_rpc_socket_path
    from ash.core import create_agent
    from ash.db import init_database
    from ash.providers.telegram import TelegramProvider
    from ash.rpc import (
        RPCServer,
        register_config_methods,
        register_log_methods,
        register_memory_methods,
    )
    from ash.server.app import create_app
    from ash.service.pid import write_pid_file

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
    components = await create_agent(
        config=ash_config,
        workspace=workspace,
        db=database,
        model_alias="default",
    )
    agent = components.agent

    # Log sandbox configuration
    from ash.service.runtime import (
        create_runtime_state_from_config,
        write_runtime_state,
    )

    sandbox = ash_config.sandbox
    logger.info(
        f"Sandbox: image={sandbox.image}, network={sandbox.network_mode}, "
        f"runtime={sandbox.runtime}, workspace={workspace.path} ({sandbox.workspace_access})"
    )

    # Write runtime state for service status
    runtime_state = create_runtime_state_from_config(ash_config, workspace.path)
    write_runtime_state(runtime_state)

    # Run memory garbage collection on startup if enabled
    if ash_config.memory.auto_gc and components.memory_manager:
        logger.debug("Running memory garbage collection")
        gc_result = await components.memory_manager.gc()
        if gc_result.removed_count > 0:
            logger.info(f"Cleaned up {gc_result.removed_count} memories")

    # Start RPC server for sandbox communication
    rpc_server: RPCServer | None = None
    if components.memory_manager:
        rpc_socket_path = get_rpc_socket_path()
        rpc_server = RPCServer(rpc_socket_path)
        register_memory_methods(
            rpc_server, components.memory_manager, components.person_manager
        )
        register_config_methods(rpc_server, ash_config, components.skill_registry)
        register_log_methods(rpc_server, get_logs_path())
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
            passive_config=ash_config.telegram.passive,
        )

    # Set up schedule watcher
    from ash.config.paths import get_schedule_file
    from ash.events import ScheduledTaskHandler, ScheduleWatcher

    schedule_file = get_schedule_file()
    schedule_watcher = ScheduleWatcher(schedule_file, timezone=ash_config.timezone)

    # Build sender map from available providers
    senders: dict[str, Any] = {}
    registrars: dict[str, Any] = {}
    if telegram_provider:
        senders["telegram"] = telegram_provider.send_message

        # Create registrar for tracking scheduled messages in thread index
        from ash.chats import ChatStateManager, ThreadIndex

        async def telegram_registrar(chat_id: str, message_id: str) -> None:
            """Register a scheduled message in the thread index for reply tracking."""
            manager = ChatStateManager(provider="telegram", chat_id=chat_id)
            thread_index = ThreadIndex(manager)
            # Scheduled messages start new threads (message_id is both external_id and thread_id)
            thread_index.register_message(message_id, message_id)

        registrars["telegram"] = telegram_registrar

    # Create and register handler if we have senders
    if senders:
        schedule_handler = ScheduledTaskHandler(
            agent, senders, registrars, timezone=ash_config.timezone
        )
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
        tool_registry=components.tool_registry,
        llm_provider=components.llm,
        memory_manager=components.memory_manager,
        memory_extractor=components.memory_extractor,
        agent_executor=components.agent_executor,
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
        shutdown_count = 0

        def handle_signal():
            nonlocal shutdown_count
            shutdown_count += 1

            if shutdown_count == 1:
                # First signal: graceful shutdown
                logger.info("Shutting down gracefully...")
                server.should_exit = True
                shutdown_event.set()
                # Stop telegram polling before cancelling task
                if telegram_provider:
                    loop.call_soon(
                        lambda: asyncio.create_task(telegram_provider.stop())
                    )
                # Cancel telegram task after stop is scheduled
                if telegram_task and not telegram_task.done():
                    telegram_task.cancel()
            else:
                # Second signal: force immediate exit
                logger.warning("Forcing immediate shutdown...")
                import os

                os._exit(1)

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
            # return_exceptions=True ensures we wait for server to finish graceful
            # shutdown after telegram is cancelled, avoiding double Ctrl+C
            await asyncio.gather(server.serve(), telegram_task, return_exceptions=True)
        else:
            await server.serve()
    finally:
        await _cleanup_server(
            schedule_watcher,
            telegram_provider,
            rpc_server,
            components.sandbox_executor,
            pid_path,
        )


async def _cleanup_server(
    schedule_watcher,
    telegram_provider,
    rpc_server,
    sandbox_executor,
    pid_path: Path,
) -> None:
    """Clean up server resources."""
    from ash.service.pid import remove_pid_file
    from ash.service.runtime import remove_runtime_state

    cleanup_timeout = 5.0  # Max seconds per cleanup operation

    for resource, method in [
        (schedule_watcher, "stop"),
        (telegram_provider, "stop"),
        (rpc_server, "stop"),
        (sandbox_executor, "cleanup"),
    ]:
        if resource:
            try:
                await asyncio.wait_for(
                    getattr(resource, method)(), timeout=cleanup_timeout
                )
            except TimeoutError:
                logger.warning(f"Cleanup {method} timed out after {cleanup_timeout}s")
            except Exception as e:
                logger.warning(f"Error during {method}: {e}")

    remove_pid_file(pid_path)
    remove_runtime_state()
