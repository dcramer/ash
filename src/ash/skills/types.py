"""Skill definitions and data types."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SkillSourceType(Enum):
    """Source type for a skill definition.

    Loading precedence (later overrides earlier):
    1. bundled - Built-in skills (lowest priority)
    2. integration - Integration-provided skills
    3. installed - Externally installed (github repos, local symlinks)
    4. user - User skills (~/.ash/skills/)
    5. workspace - Workspace skills (highest priority)
    """

    BUNDLED = "bundled"
    INTEGRATION = "integration"
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

    # Opt-in flag (bundled skills can require explicit enablement)
    opt_in: bool = False  # If True, requires [skills.<name>] enabled = true

    # Access control metadata
    sensitive: bool = False  # If True, defaults to DM-only unless chat types set
    allowed_chat_types: list[str] = field(
        default_factory=list
    )  # Empty = all chat types
    capabilities: list[str] = field(
        default_factory=list
    )  # Required namespaced capabilities (e.g. "gog.email")

    # Subagent execution settings
    env: list[str] = field(default_factory=list)  # Env vars to inject from config
    packages: list[str] = field(default_factory=list)  # System packages (apt)
    allowed_tools: list[str] = field(
        default_factory=list
    )  # Tool whitelist (empty = all)
    model: str | None = None  # Model alias override
    max_iterations: int = 10  # Iteration limit

    # Optional metadata fields
    license: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_sandbox_skill_dir(
    skill: SkillDefinition, mount_prefix: str = "/ash"
) -> str | None:
    """Derive the sandbox container path for a skill's directory.

    Computed from source_type + skill_path at prompt-build time — no absolute
    paths are stored on the data model.  Returns None for sources that are not
    mounted inside the sandbox (installed, user).
    """
    if skill.source_type == SkillSourceType.BUNDLED:
        return f"{mount_prefix}/skills/{skill.name}"
    if skill.source_type == SkillSourceType.INTEGRATION:
        if skill.skill_path is not None:
            contributor = skill.skill_path.parent.name
            return f"{mount_prefix}/integrations/{contributor}/skills/{skill.name}"
        return None
    if skill.source_type == SkillSourceType.WORKSPACE:
        return f"/workspace/skills/{skill.name}"
    return None
