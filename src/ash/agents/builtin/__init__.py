"""Built-in agents shipped with Ash."""

from ash.agents.builtin.claude_code import ClaudeCodeAgent
from ash.agents.builtin.debug_self import DebugAgent
from ash.agents.builtin.plan import PlanAgent
from ash.agents.builtin.task import TaskAgent

__all__ = [
    "ClaudeCodeAgent",
    "DebugAgent",
    "PlanAgent",
    "TaskAgent",
]


def register_builtin_agents(registry, mount_prefix: str = "/ash") -> None:
    """Register all built-in agents.

    Args:
        registry: AgentRegistry to register agents with.
        mount_prefix: Sandbox mount prefix for agents that reference container paths.

    Note:
        ClaudeCodeAgent is not registered here - it's invoked as a skill
        via use_skill(skill="claude-code", ...) instead.
    """
    registry.register(DebugAgent(mount_prefix=mount_prefix))
    registry.register(PlanAgent())
    registry.register(TaskAgent())
