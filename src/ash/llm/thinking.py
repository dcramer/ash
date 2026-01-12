"""Extended thinking configuration for Claude models.

Claude's extended thinking feature allows the model to "think out loud"
before generating a response, which can improve reasoning quality for
complex tasks.

Budget levels:
- off: No extended thinking (default)
- minimal: 1K tokens - Quick verification
- low: 4K tokens - Simple reasoning
- medium: 16K tokens - Moderate complexity
- high: 64K tokens - Complex multi-step reasoning
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ThinkingLevel(str, Enum):
    """Thinking budget levels."""

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Budget tokens for each level
THINKING_BUDGETS = {
    ThinkingLevel.OFF: 0,
    ThinkingLevel.MINIMAL: 1024,
    ThinkingLevel.LOW: 4096,
    ThinkingLevel.MEDIUM: 16384,
    ThinkingLevel.HIGH: 65536,
}


@dataclass
class ThinkingConfig:
    """Configuration for extended thinking.

    Example usage:
        # Use a preset level
        config = ThinkingConfig(level=ThinkingLevel.MEDIUM)

        # Or specify exact budget
        config = ThinkingConfig(level=ThinkingLevel.HIGH, budget_tokens=32000)

        # Disable thinking
        config = ThinkingConfig.disabled()

        # Get API parameters
        api_params = config.to_api_params()
    """

    level: ThinkingLevel = ThinkingLevel.OFF
    budget_tokens: int | None = None  # Override budget for level

    def __post_init__(self):
        """Calculate budget if not explicitly set."""
        if self.budget_tokens is None and self.level != ThinkingLevel.OFF:
            self.budget_tokens = THINKING_BUDGETS.get(self.level, 0)

    @property
    def enabled(self) -> bool:
        """Check if thinking is enabled."""
        return self.level != ThinkingLevel.OFF and (self.budget_tokens or 0) > 0

    @property
    def effective_budget(self) -> int:
        """Get the effective budget tokens."""
        if not self.enabled:
            return 0
        return self.budget_tokens or THINKING_BUDGETS.get(self.level, 0)

    def to_api_params(self) -> dict[str, Any] | None:
        """Convert to API parameters for Anthropic.

        Returns:
            Dict with thinking configuration, or None if disabled.
        """
        if not self.enabled:
            return None

        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": self.effective_budget,
            }
        }

    @classmethod
    def disabled(cls) -> "ThinkingConfig":
        """Create a disabled thinking config."""
        return cls(level=ThinkingLevel.OFF)

    @classmethod
    def from_level(cls, level: str | ThinkingLevel) -> "ThinkingConfig":
        """Create config from level string or enum.

        Args:
            level: Level name (e.g., "medium") or ThinkingLevel enum.

        Returns:
            Configured ThinkingConfig.
        """
        if isinstance(level, str):
            level = ThinkingLevel(level.lower())
        return cls(level=level)

    @classmethod
    def from_budget(cls, budget_tokens: int) -> "ThinkingConfig":
        """Create config with specific budget.

        Args:
            budget_tokens: Number of tokens for thinking.

        Returns:
            Configured ThinkingConfig.
        """
        # Find closest level
        for level, budget in sorted(
            THINKING_BUDGETS.items(), key=lambda x: x[1], reverse=True
        ):
            if budget <= budget_tokens:
                return cls(level=level, budget_tokens=budget_tokens)
        return cls.disabled()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "level": self.level.value,
            "budget_tokens": self.budget_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThinkingConfig":
        """Create from dictionary."""
        return cls(
            level=ThinkingLevel(data.get("level", "off")),
            budget_tokens=data.get("budget_tokens"),
        )


ThinkingParam = ThinkingConfig | ThinkingLevel | str | int | None


def resolve_thinking(param: ThinkingParam) -> ThinkingConfig:
    """Resolve various thinking parameter formats to ThinkingConfig.

    Args:
        param: Thinking configuration in various formats:
            - ThinkingConfig: Used directly
            - ThinkingLevel: Converted to config
            - str: Level name ("off", "minimal", etc.)
            - int: Budget tokens
            - None: Disabled

    Returns:
        ThinkingConfig instance.
    """
    if param is None:
        return ThinkingConfig.disabled()
    if isinstance(param, ThinkingConfig):
        return param
    if isinstance(param, ThinkingLevel | str):
        return ThinkingConfig.from_level(param)
    if isinstance(param, int):
        return ThinkingConfig.from_budget(param)
    return ThinkingConfig.disabled()
