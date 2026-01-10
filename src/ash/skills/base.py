"""Skill definitions and data types."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillDefinition:
    """Skill loaded from YAML."""

    name: str
    description: str
    instructions: str
    preferred_model: str | None = None
    required_tools: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 5


@dataclass
class SkillContext:
    """Context passed to skill execution."""

    session_id: str | None = None
    user_id: str | None = None
    chat_id: str | None = None
    input_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    """Result from skill execution."""

    content: str
    is_error: bool = False
    iterations: int = 0

    @classmethod
    def success(cls, content: str, iterations: int = 0) -> "SkillResult":
        """Create a successful result."""
        return cls(content=content, is_error=False, iterations=iterations)

    @classmethod
    def error(cls, message: str) -> "SkillResult":
        """Create an error result."""
        return cls(content=message, is_error=True, iterations=0)
