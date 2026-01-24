"""Built-in agents shipped with Ash."""

from ash.agents.builtin.claude_code import ClaudeCodeAgent
from ash.agents.builtin.plan import PlanAgent
from ash.agents.builtin.research import ResearchAgent
from ash.agents.builtin.skill_writer import SkillWriterAgent

__all__ = [
    "ClaudeCodeAgent",
    "PlanAgent",
    "ResearchAgent",
    "SkillWriterAgent",
]


def register_builtin_agents(registry) -> None:
    """Register all built-in agents.

    Args:
        registry: AgentRegistry to register agents with.

    Note:
        ClaudeCodeAgent is not registered here - it's invoked as a skill
        via use_skill(skill="claude-code", ...) instead.
    """
    registry.register(PlanAgent())
    registry.register(ResearchAgent())
    registry.register(SkillWriterAgent())
