"""Agent base class."""

from abc import ABC, abstractmethod

from ash.agents.types import AgentConfig, AgentContext, AgentResult


class Agent(ABC):
    """Base class for built-in agents."""

    @property
    @abstractmethod
    def config(self) -> AgentConfig: ...

    def build_system_prompt(self, context: AgentContext) -> str:
        return self.config.system_prompt

    async def execute_passthrough(
        self,
        message: str,
        context: AgentContext,
        model: str | None = None,
    ) -> AgentResult:
        """Execute passthrough agent logic.

        Passthrough agents bypass the LLM loop and run external processes directly.
        Override this method for agents with `is_passthrough=True` in their config.

        Args:
            message: The input message/task for the agent.
            context: Execution context.
            model: Optional model override from config.

        Returns:
            AgentResult with the execution result.
        """
        raise NotImplementedError(
            f"Agent '{self.config.name}' has is_passthrough=True but doesn't "
            "implement execute_passthrough()"
        )
