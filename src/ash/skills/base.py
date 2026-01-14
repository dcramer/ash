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

    # Sandbox package requirements (installed at runtime)
    apt_packages: list[str] = field(default_factory=list)  # System packages
    python_packages: list[str] = field(default_factory=list)  # Python libraries
    python_tools: list[str] = field(default_factory=list)  # CLI tools (via uvx)

    def check(self) -> tuple[bool, str | None]:
        """Check if all requirements are met.

        Note: Package requirements are not checked here - they are
        installed at container creation time via setup_command.

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


@dataclass
class SkillDefinition:
    """Skill definition - loaded from SKILL.md files.

    Skills are reusable instructions that the agent reads and follows.
    No execution happens - the agent just reads the file.
    """

    name: str
    description: str
    instructions: str
    required_tools: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    requires: SkillRequirements = field(default_factory=SkillRequirements)

    # Path to skill directory (for {baseDir} substitution)
    skill_path: Path | None = None

    def is_available(self) -> tuple[bool, str | None]:
        """Check if this skill is available on the current system.

        Returns:
            Tuple of (is_available, reason_if_not).
        """
        return self.requires.check()
