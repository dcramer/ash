"""Built-in agents shipped with Ash."""

from ash.agents.builtin.plan import PlanAgent
from ash.agents.builtin.task import TaskAgent

__all__ = [
    "PlanAgent",
    "TaskAgent",
]


def register_builtin_agents(registry) -> None:
    """Register all built-in agents."""
    registry.register(PlanAgent())
    registry.register(TaskAgent())
