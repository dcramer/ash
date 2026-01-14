"""Agent base classes and data types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    """Configuration for a built-in agent."""

    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] = field(default_factory=list)
    max_iterations: int = 10
    model: str | None = None
    is_skill_agent: bool = False


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
        return cls(content=content, iterations=iterations)

    @classmethod
    def error(cls, message: str) -> "AgentResult":
        return cls(content=message, is_error=True)


class Agent(ABC):
    """Base class for built-in agents."""

    @property
    @abstractmethod
    def config(self) -> AgentConfig: ...

    def build_system_prompt(self, context: AgentContext) -> str:
        return self.config.system_prompt
