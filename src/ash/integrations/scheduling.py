"""Scheduling integration contributor.

Spec contract: specs/subsystems.md (Integration Hooks).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ash.integrations.runtime import IntegrationContext, IntegrationContributor

if TYPE_CHECKING:
    from ash.agents import AgentExecutor
    from ash.scheduling.handler import (
        MessagePersister,
        MessageRegistrar,
        MessageSender,
    )


class SchedulingIntegration(IntegrationContributor):
    """Owns scheduling storage, RPC wiring, and optional dispatch lifecycle."""

    name = "scheduling"
    priority = 300

    def __init__(
        self,
        graph_dir: Path,
        *,
        timezone: str = "UTC",
        senders: dict[str, MessageSender] | None = None,
        registrars: dict[str, MessageRegistrar] | None = None,
        persisters: dict[str, MessagePersister] | None = None,
        agent_executor: AgentExecutor | None = None,
    ) -> None:
        self._graph_dir = graph_dir
        self._timezone = timezone
        self._senders = senders or {}
        self._registrars = registrars or {}
        self._persisters = persisters or {}
        self._agent_executor = agent_executor
        self._store = None
        self._watcher = None

    @property
    def store(self):
        return self._store

    async def setup(self, context: IntegrationContext) -> None:
        from ash.scheduling import ScheduledTaskHandler, ScheduleStore, ScheduleWatcher

        self._store = ScheduleStore(self._graph_dir)
        if self._senders:
            self._watcher = ScheduleWatcher(self._store, timezone=self._timezone)
            schedule_handler = ScheduledTaskHandler(
                context.components.agent,
                self._senders,
                self._registrars,
                self._persisters,
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
