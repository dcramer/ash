"""Skill definitions and data types."""

import os
import platform
import shutil
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillRequirements:
    """Requirements for a skill to be available.

    Skills are filtered out if requirements aren't met.
    """

    # Required binaries (all must exist in PATH)
    bins: list[str] = field(default_factory=list)

    # Required environment variables (all must be set)
    env: list[str] = field(default_factory=list)

    # Supported operating systems (empty = all)
    # Values: "darwin", "linux", "windows"
    os: list[str] = field(default_factory=list)

    def check(self) -> tuple[bool, str | None]:
        """Check if all requirements are met.

        Returns:
            Tuple of (is_met, error_message).
            If is_met is True, error_message is None.
        """
        # Check OS
        if self.os:
            current_os = platform.system().lower()
            if current_os not in self.os:
                return False, f"Requires OS: {', '.join(self.os)} (current: {current_os})"

        # Check binaries
        for bin_name in self.bins:
            if not shutil.which(bin_name):
                return False, f"Requires binary: {bin_name}"

        # Check environment variables
        for env_var in self.env:
            if not os.environ.get(env_var):
                return False, f"Requires environment variable: {env_var}"

        return True, None


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
    requires: SkillRequirements = field(default_factory=SkillRequirements)

    def is_available(self) -> tuple[bool, str | None]:
        """Check if this skill is available on the current system.

        Returns:
            Tuple of (is_available, reason_if_not).
        """
        return self.requires.check()


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
