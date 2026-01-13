"""Agent base classes and data types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    """Configuration for a built-in agent.

    Agents are code-defined subagents that run in isolated LLM loops
    with their own system prompts and tool restrictions.
    """

    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] = field(default_factory=list)  # Empty = all tools
    max_iterations: int = 10
    model: str | None = None  # None = use session model


@dataclass
class AgentContext:
    """Context passed to agent execution."""

    session_id: str | None = None
    user_id: str | None = None
    chat_id: str | None = None
    input_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from agent execution."""

    content: str
    is_error: bool = False
    iterations: int = 0

    @classmethod
    def success(cls, content: str, iterations: int = 0) -> "AgentResult":
        """Create a successful result."""
        return cls(content=content, iterations=iterations)

    @classmethod
    def error(cls, message: str) -> "AgentResult":
        """Create an error result."""
        return cls(content=message, is_error=True)


class Agent(ABC):
    """Base class for built-in agents.

    Agents are autonomous subprocesses that run isolated LLM loops
    for complex multi-step tasks. They have their own:
    - System prompt
    - Tool restrictions (can whitelist specific tools)
    - Max iterations limit
    - Optional model override

    Unlike skills (which are markdown files the main agent reads),
    agents execute in their own context and return results.
    """

    @property
    @abstractmethod
    def config(self) -> AgentConfig:
        """Return agent configuration."""
        ...

    def build_system_prompt(self, context: AgentContext) -> str:
        """Build system prompt, optionally injecting context.

        Override this method to customize the system prompt
        based on the execution context.

        Args:
            context: Execution context with session info and input data.

        Returns:
            System prompt string.
        """
        return self.config.system_prompt
