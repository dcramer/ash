"""Server command for running the Ash service."""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.skills import SkillRegistry


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
    from ash.cli.runtime import bootstrap_runtime
    from ash.logging import configure_logging

    # Configure logging with Rich for colorful server output and file logging
    configure_logging(use_rich=True, log_to_file=True)

    from ash.config import load_config
    from ash.config.paths import (
        get_logs_path,
        get_pid_path,
        get_rpc_socket_path,
        get_schedule_file,
        get_sessions_path,
    )
    from ash.integrations import (
        active_integrations,
        active_rpc_server,
        create_default_integrations,
    )
    from ash.providers import build_provider_runtime
    from ash.server import ServerRunner, create_app
    from ash.service.pid import write_pid_file

    # Write PID file for service management
    pid_path = get_pid_path()
    write_pid_file(pid_path)

    # Load configuration
    logger.info("config_loading")
    ash_config = load_config(config_path)

    logger.info("workspace_loading")
    logger.info("agent_setup")
    runtime = await bootstrap_runtime(
        config=ash_config,
        model_alias="default",
        initialize_sentry=True,
        sentry_server_mode=True,
    )
    if runtime.sentry_initialized:
        logger.info("sentry_initialized")
    components = runtime.components
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
            "sandbox.workspace_path": str(runtime.workspace.path),
            "sandbox.workspace_access": sandbox.workspace_access,
        },
    )

    # Write runtime state for service status
    runtime_state = create_runtime_state_from_config(ash_config, runtime.workspace.path)
    write_runtime_state(runtime_state)

    # Run memory garbage collection on startup if enabled
    if ash_config.memory.auto_gc and components.memory_manager:
        logger.debug("Running memory garbage collection")
        gc_result = await components.memory_manager.gc()
        if gc_result.removed_count > 0:
            logger.info(
                "memory_gc_complete", extra={"memory.count": gc_result.removed_count}
            )

    provider_runtime = build_provider_runtime(ash_config)
    telegram_provider = provider_runtime.telegram_provider
    if telegram_provider:
        logger.info("telegram_provider_setup")

    if not provider_runtime.senders:
        logger.debug("schedule watcher disabled (no providers)")

    skill_auto_sync_task: asyncio.Task[None] | None = None
    if ash_config.skill_auto_sync and ash_config.skill_sources:
        skill_auto_sync_task = _start_skill_auto_sync_task(
            ash_config, components.skill_registry
        )

    # Compose integration contributors for runtime wiring.
    default_integrations = create_default_integrations(
        mode="serve",
        logs_path=get_logs_path(),
        schedule_file=get_schedule_file(),
        include_memory=True,
        timezone=ash_config.timezone,
        senders=provider_runtime.senders,
        registrars=provider_runtime.registrars,
        persisters=provider_runtime.persisters,
        agent_executor=components.agent_executor,
    )
    async with active_integrations(
        config=ash_config,
        components=components,
        mode="serve",
        sessions_path=get_sessions_path(),
        contributors=default_integrations.contributors,
    ) as (integration_runtime, integration_context):
        if default_integrations.scheduling is None:
            raise RuntimeError("schedule integration setup missing")
        if default_integrations.scheduling.store is None:
            raise RuntimeError("schedule integration setup failed")
        logger.debug(
            f"Schedule store: {default_integrations.scheduling.store.schedule_file}"
        )

        try:
            rpc_socket_path = get_rpc_socket_path()
            async with active_rpc_server(
                runtime=integration_runtime,
                context=integration_context,
                socket_path=rpc_socket_path,
            ):
                logger.info(
                    "rpc_server_started", extra={"socket.path": str(rpc_socket_path)}
                )

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
                logger.info(
                    "server_starting",
                    extra={"server.address": host, "server.port": port},
                )

                runner = ServerRunner(
                    fastapi_app,
                    host=host,
                    port=port,
                    telegram_provider=telegram_provider,
                )
                await runner.run()
        finally:
            await _cleanup_server(
                telegram_provider,
                components.sandbox_executor,
                pid_path,
                skill_auto_sync_task,
            )


def _start_skill_auto_sync_task(
    ash_config: "AshConfig",
    skill_registry: "SkillRegistry | None" = None,
) -> asyncio.Task[None]:
    """Start periodic background sync of configured skill sources."""
    from ash.skills.installer import SkillInstaller

    interval_minutes = max(1, int(ash_config.skill_update_interval_minutes))
    logger.info(
        "skill_sources_auto_sync_started",
        extra={"interval.minutes": interval_minutes},
    )

    async def _loop() -> None:
        installer = SkillInstaller()
        failure_streak = 0
        while True:
            try:
                report = await asyncio.to_thread(
                    installer.sync_all_report, ash_config.skill_sources
                )
                if skill_registry is not None and report.changed:
                    skill_registry.reload_all(ash_config.workspace)
                failure_count = len(report.failed)
                synced_count = len(report.synced)
                changed_count = len(report.changed)
                if failure_count:
                    failure_streak += 1
                else:
                    failure_streak = 0
                logger.info(
                    "skill_sources_auto_sync_complete",
                    extra={
                        "count": synced_count,
                        "changed_count": changed_count,
                        "failed_count": failure_count,
                    },
                )
                if report.failed:
                    logger.warning(
                        "skill_sources_auto_sync_partial_failure",
                        extra={
                            "failed_count": failure_count,
                            "failed_sources": [
                                s.repo or str(s.path) for s, _ in report.failed
                            ],
                        },
                    )
            except Exception as e:
                failure_streak += 1
                logger.warning(
                    "skill_sources_auto_sync_failed",
                    extra={"error.message": str(e)},
                )
            base_sleep = interval_minutes * 60
            if failure_streak <= 0:
                sleep_seconds = base_sleep
            else:
                sleep_seconds = min(base_sleep * (2 ** min(failure_streak, 4)), 3600)
            await asyncio.sleep(sleep_seconds)

    return asyncio.create_task(_loop(), name="skill-auto-sync")


async def _cleanup_server(
    telegram_provider,
    sandbox_executor,
    pid_path: Path,
    skill_auto_sync_task: asyncio.Task[None] | None = None,
) -> None:
    """Clean up server resources."""
    from ash.service.pid import remove_pid_file
    from ash.service.runtime import remove_runtime_state

    cleanup_timeout = 5.0  # Max seconds per cleanup operation

    for resource, method in [
        (telegram_provider, "stop"),
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

    if skill_auto_sync_task:
        skill_auto_sync_task.cancel()
        try:
            await skill_auto_sync_task
        except asyncio.CancelledError:
            pass

    remove_pid_file(pid_path)
    remove_runtime_state()
