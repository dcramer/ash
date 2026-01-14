"""Skill definitions and data types."""

import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillRequirements:
    """Requirements for a skill to be available.

    Skills are filtered out if requirements aren't met.
    """

    # Required binaries (all must exist in PATH)
    bins: list[str] = field(default_factory=list)

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

        return True, None


@dataclass
class SkillDefinition:
    """Skill definition - loaded from SKILL.md files.

    Skills are invoked via the use_skill tool and run as subagents
    with isolated sessions and scoped environments.
    """

    name: str
    description: str
    instructions: str

    # Availability filtering
    requires: SkillRequirements = field(default_factory=SkillRequirements)
    skill_path: Path | None = None  # Path to skill directory

    # Subagent execution settings
    env: list[str] = field(default_factory=list)  # Env vars to inject from config
    allowed_tools: list[str] = field(
        default_factory=list
    )  # Tool whitelist (empty = all)
    model: str | None = None  # Model alias override
    max_iterations: int = 10  # Iteration limit

    def is_available(self) -> tuple[bool, str | None]:
        """Check if this skill is available on the current system.

        Returns:
            Tuple of (is_available, reason_if_not).
        """
        return self.requires.check()
