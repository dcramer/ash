"""Skill definitions and data types."""

import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path
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
                return (
                    False,
                    f"Requires OS: {', '.join(self.os)} (current: {current_os})",
                )

        # Check binaries
        for bin_name in self.bins:
            if not shutil.which(bin_name):
                return False, f"Requires binary: {bin_name}"

        # Check environment variables
        for env_var in self.env:
            if not os.environ.get(env_var):
                return False, f"Requires environment variable: {env_var}"

        return True, None


# Type alias for dynamic skill config builder
# Signature: (input_data: dict, **kwargs) -> SubagentConfig
ConfigBuilder = Any  # Callable[[dict[str, Any], ...], "SubagentConfig"]


@dataclass
class SkillDefinition:
    """Skill definition - can be loaded from SKILL.md or registered dynamically."""

    name: str
    description: str
    instructions: str = ""  # Empty for dynamic skills
    subagent: bool = False  # True = isolated LLM loop, False = inline instructions
    model: str | None = None  # Model alias (e.g., "default", "sonnet")
    required_tools: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 5
    requires: SkillRequirements = field(default_factory=SkillRequirements)

    # Config: list of env var names with optional =default suffix
    # e.g., ["API_KEY", "TIMEOUT=30"]
    config: list[str] = field(default_factory=list)

    # Resolved config values (populated by registry)
    config_values: dict[str, str] = field(default_factory=dict)

    # Path to skill directory (for loading config.toml)
    skill_path: Path | None = None

    # For dynamic skills: callable that builds SubagentConfig from input
    # If set, this skill is dynamic and uses subagent execution
    build_config: ConfigBuilder | None = None

    @property
    def is_dynamic(self) -> bool:
        """Check if this is a dynamic skill (has build_config)."""
        return self.build_config is not None

    @staticmethod
    def parse_config_spec(spec: str) -> tuple[str, str | None]:
        """Parse config spec into (name, default_or_none).

        Args:
            spec: Config spec string, e.g. "API_KEY" or "TIMEOUT=30".

        Returns:
            Tuple of (name, default) where default is None if not specified.
        """
        if "=" in spec:
            name, default = spec.split("=", 1)
            return name.strip(), default.strip()
        return spec.strip(), None

    def is_available(self) -> tuple[bool, str | None]:
        """Check if this skill is available on the current system.

        Returns:
            Tuple of (is_available, reason_if_not).
        """
        # Check system requirements first
        ok, msg = self.requires.check()
        if not ok:
            return ok, msg

        # Check config requirements
        return self.is_config_valid()

    def is_config_valid(self) -> tuple[bool, str | None]:
        """Check if all required config values are present.

        Returns:
            Tuple of (is_valid, error_message).
        """
        for item in self.config:
            name, default = self.parse_config_spec(item)
            if default is not None:
                # Has default, so not required
                continue
            if name not in self.config_values:
                return False, f"Missing required config: {name}"
        return True, None

    def get_config_defaults(self) -> dict[str, str]:
        """Get default values from config declarations.

        Returns:
            Dict of name -> default value for items with defaults.
        """
        defaults = {}
        for item in self.config:
            name, default = self.parse_config_spec(item)
            if default is not None:
                defaults[name] = default
        return defaults

    def get_config_names(self) -> list[str]:
        """Get list of config variable names.

        Returns:
            List of config names without defaults.
        """
        return [self.parse_config_spec(item)[0] for item in self.config]


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
    # Environment variables to inject into bash (for inline skills with config)
    skill_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        content: str,
        iterations: int = 0,
        skill_env: dict[str, str] | None = None,
    ) -> "SkillResult":
        """Create a successful result."""
        return cls(
            content=content,
            is_error=False,
            iterations=iterations,
            skill_env=skill_env or {},
        )

    @classmethod
    def error(cls, message: str) -> "SkillResult":
        """Create an error result."""
        return cls(content=message, is_error=True, iterations=0)


@dataclass
class SubagentConfig:
    """Configuration for running a subagent.

    Subagents are isolated LLM loops with their own context and tool access.
    Used by:
    - Skills with subagent=True
    - Tools that spawn subagents (e.g., write_skill)
    """

    # System prompt for the subagent
    system_prompt: str

    # Tools the subagent can use (empty = all available)
    allowed_tools: list[str] = field(default_factory=list)

    # Maximum LLM iterations before stopping
    max_iterations: int = 10

    # Model alias to use (None = use default)
    model: str | None = None

    # Environment variables to pass to tool execution
    env: dict[str, str] = field(default_factory=dict)

    # Initial user message to start the conversation
    initial_message: str = "Execute according to the instructions provided."
