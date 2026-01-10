"""Abstract tool interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolContext:
    """Context passed to tool execution."""

    session_id: str | None = None
    user_id: str | None = None
    chat_id: str | None = None
    provider: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result from tool execution."""

    content: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, content: str, **metadata: Any) -> "ToolResult":
        """Create a successful result."""
        return cls(content=content, is_error=False, metadata=metadata)

    @classmethod
    def error(cls, message: str, **metadata: Any) -> "ToolResult":
        """Create an error result."""
        return cls(content=message, is_error=True, metadata=metadata)


class Tool(ABC):
    """Abstract base class for tools.

    Tools are capabilities that the agent can use to interact with
    external systems, execute code, search the web, etc.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the LLM."""
        ...

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for tool input parameters."""
        ...

    @abstractmethod
    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the tool with the given input.

        Args:
            input_data: Tool input matching the input_schema.
            context: Execution context.

        Returns:
            Tool execution result.
        """
        ...

    def to_definition(self) -> dict[str, Any]:
        """Convert to LLM tool definition format.

        Returns:
            Dict suitable for LLM tool definitions.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
