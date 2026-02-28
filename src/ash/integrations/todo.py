"""Todo integration contributor.

Spec contract: specs/subsystems.md (Integration Hooks), specs/todos.md.
"""

from __future__ import annotations

from pathlib import Path

from ash.integrations.runtime import IntegrationContext, IntegrationContributor
from ash.todos import TodoManager, create_todo_manager


class TodoIntegration(IntegrationContributor):
    """Owns todo manager setup and RPC surface registration."""

    name = "todo"
    priority = 250

    def __init__(self, *, graph_dir: Path, schedule_enabled: bool = False) -> None:
        self._graph_dir = graph_dir
        self._schedule_enabled = schedule_enabled
        self._manager: TodoManager | None = None

    @property
    def manager(self) -> TodoManager | None:
        return self._manager

    async def setup(self, context: IntegrationContext) -> None:
        store = context.components.memory_manager
        if store is None:
            self._manager = await create_todo_manager(self._graph_dir)
            return
        self._manager = await create_todo_manager(
            self._graph_dir,
            graph=store.graph,
            persistence=store.persistence,
        )

    def register_rpc_methods(self, server, context: IntegrationContext) -> None:
        from ash.rpc.methods.todo import register_todo_methods
        from ash.scheduling import ScheduleStore

        if self._manager is None:
            return
        schedule_store = (
            ScheduleStore(self._graph_dir) if self._schedule_enabled else None
        )
        register_todo_methods(server, self._manager, schedule_store=schedule_store)

    def augment_skill_instructions(
        self,
        skill_name: str,
        context: IntegrationContext,
    ) -> list[str]:
        if skill_name != "todo":
            return []
        if not self._schedule_enabled:
            return []
        return [
            "Scheduling is enabled. You can use `ash-sb todo remind` and `ash-sb todo unremind` to manage reminders on todos.",
            "When a user mentions a reminder or due date with a time component, offer to set a reminder as well.",
        ]
