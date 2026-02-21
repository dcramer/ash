"""Built-in integration contributors."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ash.integrations.runtime import IntegrationContext, IntegrationContributor

if TYPE_CHECKING:
    from ash.agents import AgentExecutor
    from ash.scheduling.handler import MessageRegistrar, MessageSender


class MemoryIntegration(IntegrationContributor):
    """Registers memory RPC surface when memory is enabled."""

    name = "memory"
    priority = 200

    def register_rpc_methods(self, server, context: IntegrationContext) -> None:
        from ash.rpc.methods.memory import register_memory_methods

        components = context.components
        if not components.memory_manager:
            return
        register_memory_methods(
            server,
            components.memory_manager,
            components.person_manager,
            memory_extractor=components.memory_extractor,
            sessions_path=context.sessions_path,
        )


class SchedulingIntegration(IntegrationContributor):
    """Owns scheduling storage, RPC wiring, and optional dispatch lifecycle."""

    name = "scheduling"
    priority = 300

    def __init__(
        self,
        schedule_file: Path,
        *,
        timezone: str = "UTC",
        senders: dict[str, MessageSender] | None = None,
        registrars: dict[str, MessageRegistrar] | None = None,
        agent_executor: AgentExecutor | None = None,
    ) -> None:
        self._schedule_file = schedule_file
        self._timezone = timezone
        self._senders = senders or {}
        self._registrars = registrars or {}
        self._agent_executor = agent_executor
        self._store = None
        self._watcher = None

    @property
    def store(self):
        return self._store

    async def setup(self, context: IntegrationContext) -> None:
        from ash.scheduling import ScheduledTaskHandler, ScheduleStore, ScheduleWatcher

        self._store = ScheduleStore(self._schedule_file)
        if self._senders:
            self._watcher = ScheduleWatcher(self._store, timezone=self._timezone)
            schedule_handler = ScheduledTaskHandler(
                context.components.agent,
                self._senders,
                self._registrars,
                timezone=self._timezone,
                agent_executor=self._agent_executor,
            )
            self._watcher.add_handler(schedule_handler.handle)

    def register_rpc_methods(self, server, context: IntegrationContext) -> None:
        from ash.rpc.methods.schedule import register_schedule_methods

        if self._store is None:
            return
        register_schedule_methods(server, self._store)

    async def on_startup(self, context: IntegrationContext) -> None:
        if self._watcher is not None:
            await self._watcher.start()

    async def on_shutdown(self, context: IntegrationContext) -> None:
        if self._watcher is not None:
            await self._watcher.stop()


class RuntimeRPCIntegration(IntegrationContributor):
    """Registers core runtime RPC surfaces."""

    name = "runtime-rpc"
    priority = 100

    def __init__(self, logs_path: Path) -> None:
        self._logs_path = logs_path

    def register_rpc_methods(self, server, context: IntegrationContext) -> None:
        from ash.rpc.methods.config import register_config_methods
        from ash.rpc.methods.logs import register_log_methods

        register_config_methods(
            server,
            context.config,
            context.components.skill_registry,
        )
        register_log_methods(server, self._logs_path)
