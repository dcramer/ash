"""Todo integration contributor.

Spec contract: specs/subsystems.md (Integration Hooks), specs/todos.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ash.core.prompt_keys import TOOL_ROUTING_RULES_KEY
from ash.integrations.runtime import IntegrationContext, IntegrationContributor
from ash.todos import TodoManager, create_todo_manager

if TYPE_CHECKING:
    from ash.core.prompt import PromptContext
    from ash.core.session import SessionState


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

    def augment_prompt_context(
        self,
        prompt_context: PromptContext,
        session: SessionState,
        context: IntegrationContext,
    ) -> PromptContext:
        _ = session
        _ = context
        lines = prompt_context.extra_context.setdefault(TOOL_ROUTING_RULES_KEY, [])
        if isinstance(lines, list):
            lines.append(
                "- For canonical todo list operations, use `ash-sb todo` instead of memory writes (`ash-sb todo add`, `ash-sb todo list`, `ash-sb todo done`)."
            )
            lines.append(
                "- When reporting todos to users, summarize the task text naturally and avoid exposing internal todo IDs unless the user asks for IDs or a follow-up mutation requires one."
            )
        return prompt_context
