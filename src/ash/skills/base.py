"""Skill definitions and data types."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillDefinition:
    """Skill definition - loaded from SKILL.md files.

    Skills are invoked via the use_skill tool and run as subagents
    with isolated sessions and scoped environments.
    """

    name: str
    description: str
    instructions: str

    skill_path: Path | None = None  # Path to skill directory

    # Provenance
    authors: list[str] = field(default_factory=list)  # Who created/maintains this skill
    rationale: str | None = None  # Why this skill was created

    # Subagent execution settings
    env: list[str] = field(default_factory=list)  # Env vars to inject from config
    packages: list[str] = field(default_factory=list)  # System packages (apt)
    allowed_tools: list[str] = field(
        default_factory=list
    )  # Tool whitelist (empty = all)
    model: str | None = None  # Model alias override
    max_iterations: int = 10  # Iteration limit
