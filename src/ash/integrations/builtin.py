"""Built-in integration contributors."""

from __future__ import annotations

from pathlib import Path

from ash.integrations.runtime import IntegrationContext, IntegrationContributor


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
    """Owns scheduling storage and schedule RPC wiring."""

    name = "scheduling"
    priority = 300

    def __init__(self, schedule_file: Path) -> None:
        self._schedule_file = schedule_file
        self._store = None

    @property
    def store(self):
        return self._store

    async def setup(self, context: IntegrationContext) -> None:
        from ash.scheduling import ScheduleStore

        self._store = ScheduleStore(self._schedule_file)

    def register_rpc_methods(self, server, context: IntegrationContext) -> None:
        from ash.rpc.methods.schedule import register_schedule_methods

        if self._store is None:
            return
        register_schedule_methods(server, self._store)
