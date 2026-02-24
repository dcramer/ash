"""Built-in agents shipped with Ash."""

from ash.agents.builtin.task import TaskAgent

__all__ = [
    "TaskAgent",
]


def register_builtin_agents(registry) -> None:
    """Register all built-in agents."""
    registry.register(TaskAgent())
