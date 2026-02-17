"""Agent registry for managing built-in agents."""

import logging

from ash.agents.base import Agent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Registry for built-in agents."""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        name = agent.config.name
        if name in self._agents:
            logger.warning(
                "agent_already_registered", extra={"gen_ai.agent.name": name}
            )
        self._agents[name] = agent
        logger.debug(f"Registered agent: {name}")

    def get(self, name: str) -> Agent:
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not found")
        return self._agents[name]

    def list_agents(self) -> list[Agent]:
        return list(self._agents.values())

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __iter__(self):
        return iter(self._agents.values())
