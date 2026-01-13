"""Agent registry for managing built-in agents."""

import logging

from ash.agents.base import Agent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Registry for built-in agents.

    Agents are code-defined subagents that run in isolated LLM loops.
    Unlike skills (user-defined markdown files), agents are shipped
    with Ash and provide built-in capabilities like research.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        """Register an agent.

        Args:
            agent: Agent instance to register.
        """
        name = agent.config.name
        if name in self._agents:
            logger.warning(f"Agent '{name}' already registered, overwriting")
        self._agents[name] = agent
        logger.debug(f"Registered agent: {name}")

    def get(self, name: str) -> Agent:
        """Get agent by name.

        Args:
            name: Agent name.

        Returns:
            Agent instance.

        Raises:
            KeyError: If agent not found.
        """
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not found")
        return self._agents[name]

    def has(self, name: str) -> bool:
        """Check if agent exists.

        Args:
            name: Agent name.

        Returns:
            True if agent exists.
        """
        return name in self._agents

    def list_available(self) -> list[Agent]:
        """List all registered agents.

        Returns:
            List of agent instances.
        """
        return list(self._agents.values())

    def list_names(self) -> list[str]:
        """List all registered agent names.

        Returns:
            List of agent names.
        """
        return list(self._agents.keys())

    def __len__(self) -> int:
        """Get number of registered agents."""
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        """Check if agent is registered."""
        return name in self._agents

    def __iter__(self):
        """Iterate over agent instances."""
        return iter(self._agents.values())
