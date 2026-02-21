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
            asyncio.run(_run_server(config, host, port))
        except KeyboardInterrupt:
            # Use print here since logging may not be configured yet
            print("\nServer stopped")


async def _run_server(
    config_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Run the server asynchronously."""
    # Runtime harness boundary.
    # Spec contract: specs/subsystems.md (Integration Hooks).
    import signal as signal_module

    import uvicorn

    from ash.logging import configure_logging

    # Configure logging with Rich for colorful server output and file logging
    configure_logging(use_rich=True, log_to_file=True)

    from ash.config import WorkspaceLoader, load_config
    from ash.config.paths import (
        get_graph_dir,
        get_logs_path,
        get_pid_path,
        get_rpc_socket_path,
        get_schedule_file,
        get_sessions_path,
    )
    from ash.core import create_agent
    from ash.integrations import (
        MemoryIntegration,
        RuntimeRPCIntegration,
        SchedulingIntegration,
        compose_integrations,
    )
    from ash.providers.telegram import TelegramProvider
    from ash.rpc import RPCServer
    from ash.server.app import create_app
    from ash.service.pid import write_pid_file

    # Write PID file for service management
    pid_path = get_pid_path()
    write_pid_file(pid_path)

    # Load configuration
    logger.info("config_loading")
    ash_config = load_config(config_path)

    # Initialize Sentry for server mode
    if ash_config.sentry:
        from ash.observability import init_sentry

        if init_sentry(ash_config.sentry, server_mode=True):
            logger.info("sentry_initialized")

    # Set up graph directory for memory
    graph_dir = get_graph_dir()

    # Load workspace
    logger.info("workspace_loading")
    workspace_loader = WorkspaceLoader(ash_config.workspace)
    workspace_loader.ensure_workspace()
    workspace = workspace_loader.load()

    # Create agent with all dependencies
    logger.info("agent_setup")
    components = await create_agent(
        config=ash_config,
        workspace=workspace,
        graph_dir=graph_dir,
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
        "sandbox_config",
        extra={
            "sandbox.image": sandbox.image,
            "sandbox.network_mode": sandbox.network_mode,
            "sandbox.runtime": sandbox.runtime,
            "sandbox.workspace_path": str(workspace.path),
            "sandbox.workspace_access": sandbox.workspace_access,
        },
    )

    # Write runtime state for service status
    runtime_state = create_runtime_state_from_config(ash_config, workspace.path)
    write_runtime_state(runtime_state)

    # Run memory garbage collection on startup if enabled
    if ash_config.memory.auto_gc and components.memory_manager:
        logger.debug("Running memory garbage collection")
        gc_result = await components.memory_manager.gc()
        if gc_result.removed_count > 0:
            logger.info(
                "memory_gc_complete", extra={"memory.count": gc_result.removed_count}
            )

    # Set up Telegram if configured
    telegram_provider = None
    if ash_config.telegram and ash_config.telegram.bot_token:
        logger.info("telegram_provider_setup")
        telegram_provider = TelegramProvider(
            bot_token=ash_config.telegram.bot_token.get_secret_value(),
            allowed_users=ash_config.telegram.allowed_users,
            allowed_groups=ash_config.telegram.allowed_groups,
            group_mode=ash_config.telegram.group_mode,
            passive_config=ash_config.telegram.passive,
        )

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

    if not senders:
        logger.debug("schedule watcher disabled (no providers)")

    # Compose integration contributors for runtime wiring.
    schedule_integration = SchedulingIntegration(
        get_schedule_file(),
        timezone=ash_config.timezone,
        senders=senders,
        registrars=registrars,
        agent_executor=components.agent_executor,
    )
    integration_runtime, integration_context = await compose_integrations(
        config=ash_config,
        components=components,
        mode="serve",
        sessions_path=get_sessions_path(),
        contributors=[
            RuntimeRPCIntegration(get_logs_path()),
            MemoryIntegration(),
            schedule_integration,
        ],
    )

    if schedule_integration.store is None:
        raise RuntimeError("schedule integration setup failed")
    logger.debug(f"Schedule store: {schedule_integration.store.schedule_file}")

    rpc_socket_path = get_rpc_socket_path()
    rpc_server = RPCServer(rpc_socket_path)
    integration_runtime.register_rpc_methods(rpc_server, integration_context)
    await rpc_server.start()
    logger.info("rpc_server_started", extra={"socket.path": str(rpc_socket_path)})

    logger.debug(f"Tools: {', '.join(components.tool_registry.names)}")
    if components.skill_registry:
        logger.debug(f"Skills: {len(components.skill_registry)} discovered")

    # Create FastAPI app
    logger.info("server_creating")
    fastapi_app = create_app(
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
    logger.info("server_starting", extra={"server.address": host, "server.port": port})

    try:
        uvicorn_config = uvicorn.Config(
            fastapi_app,
            host=host,
            port=port,
            log_level="info",
            log_config=None,  # Use our logging config, not uvicorn's
        )
        server = uvicorn.Server(uvicorn_config)

        await integration_runtime.on_startup(integration_context)

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
                logger.info("server_shutting_down")
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
                logger.warning("server_force_shutdown")
                import os

                os._exit(1)

        for sig in (signal_module.SIGTERM, signal_module.SIGINT):
            loop.add_signal_handler(sig, handle_signal)

        if telegram_provider:
            # Run both uvicorn and telegram polling
            logger.info("telegram_polling_starting")

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
                        logger.info("telegram_polling_cancelled")
                else:
                    logger.error("telegram_handler_timeout")

            telegram_task = asyncio.create_task(start_telegram())
            # return_exceptions=True ensures we wait for server to finish graceful
            # shutdown after telegram is cancelled, avoiding double Ctrl+C
            await asyncio.gather(server.serve(), telegram_task, return_exceptions=True)
        else:
            await server.serve()
    finally:
        await integration_runtime.on_shutdown(integration_context)
        await _cleanup_server(
            telegram_provider,
            rpc_server,
            components.sandbox_executor,
            pid_path,
        )


async def _cleanup_server(
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
                logger.warning(
                    "cleanup_timed_out",
                    extra={
                        "cleanup.method": method,
                        "operation.timeout": cleanup_timeout,
                    },
                )
            except Exception as e:
                logger.warning(
                    "cleanup_error",
                    extra={"cleanup.method": method, "error.message": str(e)},
                )

    remove_pid_file(pid_path)
    remove_runtime_state()
