"""Built-in agents shipped with Ash."""

from ash.agents.builtin.research import ResearchAgent
from ash.agents.builtin.skill_writer import SkillWriterAgent

__all__ = [
    "ResearchAgent",
    "SkillWriterAgent",
]


def register_builtin_agents(registry) -> None:
    """Register all built-in agents.

    Args:
        registry: AgentRegistry to register agents with.
    """
    registry.register(ResearchAgent())
    registry.register(SkillWriterAgent())
