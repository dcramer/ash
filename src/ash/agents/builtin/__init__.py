"""Built-in agents shipped with Ash."""

from ash.agents.builtin.claude_code import ClaudeCodeAgent
from ash.agents.builtin.plan import PlanAgent
from ash.agents.builtin.task import TaskAgent

__all__ = [
    "ClaudeCodeAgent",
    "PlanAgent",
    "TaskAgent",
]


def register_builtin_agents(registry) -> None:
    """Register all built-in agents.

    Note:
        ClaudeCodeAgent is not registered here - it's invoked as a skill
        via use_skill(skill="claude-code", ...) instead.
    """
    registry.register(PlanAgent())
    registry.register(TaskAgent())
