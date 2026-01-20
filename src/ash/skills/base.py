"""Skill definitions and data types."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SkillSourceType(Enum):
    """Source type for a skill definition.

    Loading precedence (later overrides earlier):
    1. bundled - Built-in skills (lowest priority)
    2. installed - Externally installed (github repos, local symlinks)
    3. user - User skills (~/.ash/skills/)
    4. workspace - Workspace skills (highest priority)
    """

    BUNDLED = "bundled"
    INSTALLED = "installed"
    USER = "user"
    WORKSPACE = "workspace"


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

    # Source tracking
    source_type: SkillSourceType = (
        SkillSourceType.WORKSPACE
    )  # Where skill was loaded from
    source_repo: str | None = None  # GitHub repo (owner/repo) if from installed
    source_ref: str | None = None  # Git ref (branch/tag/commit) if from installed

    # Subagent execution settings
    env: list[str] = field(default_factory=list)  # Env vars to inject from config
    packages: list[str] = field(default_factory=list)  # System packages (apt)
    allowed_tools: list[str] = field(
        default_factory=list
    )  # Tool whitelist (empty = all)
    model: str | None = None  # Model alias override
    max_iterations: int = 10  # Iteration limit
