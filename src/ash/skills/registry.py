"""Skill registry for discovering and loading skills from workspace."""

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from ash.skills.base import SkillDefinition

logger = logging.getLogger(__name__)

# Regex to match YAML frontmatter: starts with ---, ends with ---
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


class SkillRegistry:
    """Registry for skill definitions.

    Discovers and loads skills from workspace/skills/.
    Supports markdown files with YAML frontmatter (preferred) or pure YAML.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._skills: dict[str, SkillDefinition] = {}

    def discover(self, workspace_path: Path) -> None:
        """Load skills from workspace/skills/.

        Supports:
        - Directory format: skills/<name>/SKILL.md (preferred)
        - Flat markdown: skills/<name>.md (convenience)
        - Pure YAML: skills/<name>.yaml (backward compatibility)

        Args:
            workspace_path: Path to workspace directory.
        """
        skills_dir = workspace_path / "skills"
        if not skills_dir.exists():
            logger.debug(f"Skills directory not found: {skills_dir}")
            return

        # Preferred: skills/<name>/SKILL.md
        for skill_dir in skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    try:
                        self._load_markdown_skill(skill_file, default_name=skill_dir.name)
                    except Exception as e:
                        logger.warning(f"Failed to load skill from {skill_file}: {e}")

        # Also support flat markdown files
        for md_file in skills_dir.glob("*.md"):
            try:
                self._load_markdown_skill(md_file)
            except Exception as e:
                logger.warning(f"Failed to load skill from {md_file}: {e}")

        # Also support pure YAML for backward compatibility
        for yaml_file in skills_dir.glob("*.yaml"):
            try:
                self._load_yaml_skill(yaml_file)
            except Exception as e:
                logger.warning(f"Failed to load skill from {yaml_file}: {e}")

        for yml_file in skills_dir.glob("*.yml"):
            try:
                self._load_yaml_skill(yml_file)
            except Exception as e:
                logger.warning(f"Failed to load skill from {yml_file}: {e}")

        logger.info(f"Discovered {len(self._skills)} skills from {skills_dir}")

    def _load_markdown_skill(
        self, path: Path, default_name: str | None = None
    ) -> None:
        """Load a skill from a markdown file with YAML frontmatter.

        Format:
            ---
            description: What the skill does
            preferred_model: fast  # optional
            ---

            Instructions go here as markdown body.

        Args:
            path: Path to markdown file.
            default_name: Default name if not in frontmatter (e.g., directory name).
        """
        content = path.read_text()

        # Parse frontmatter
        match = FRONTMATTER_PATTERN.match(content)
        if not match:
            raise ValueError("No YAML frontmatter found (must start with ---)")

        frontmatter_yaml = match.group(1)
        instructions = content[match.end() :].strip()

        data = yaml.safe_load(frontmatter_yaml)
        if not isinstance(data, dict):
            raise ValueError("Frontmatter must be a YAML mapping")

        # Name priority: frontmatter > default_name > filename stem
        name = data.get("name") or default_name or path.stem

        if "description" not in data:
            raise ValueError("Skill missing required field: description")

        if not instructions:
            raise ValueError("Skill missing instructions (markdown body)")

        skill = SkillDefinition(
            name=name,
            description=data["description"],
            instructions=instructions,
            preferred_model=data.get("preferred_model"),
            required_tools=data.get("required_tools", []),
            input_schema=data.get("input_schema", {}),
            max_iterations=data.get("max_iterations", 5),
        )

        self._skills[skill.name] = skill
        logger.debug(f"Loaded skill: {skill.name} from {path}")

    def _load_yaml_skill(self, path: Path) -> None:
        """Load a skill from a pure YAML file (backward compatibility).

        Args:
            path: Path to YAML file.
        """
        with path.open() as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid skill file: expected dict, got {type(data)}")

        # Name defaults to filename without extension
        name = data.get("name", path.stem)

        if "description" not in data:
            raise ValueError("Skill missing required field: description")
        if "instructions" not in data:
            raise ValueError("Skill missing required field: instructions")

        skill = SkillDefinition(
            name=name,
            description=data["description"],
            instructions=data["instructions"],
            preferred_model=data.get("preferred_model"),
            required_tools=data.get("required_tools", []),
            input_schema=data.get("input_schema", {}),
            max_iterations=data.get("max_iterations", 5),
        )

        self._skills[skill.name] = skill
        logger.debug(f"Loaded skill: {skill.name} from {path}")

    def register(self, skill: SkillDefinition) -> None:
        """Register a skill directly.

        Args:
            skill: Skill definition to register.
        """
        self._skills[skill.name] = skill
        logger.debug(f"Registered skill: {skill.name}")

    def get(self, name: str) -> SkillDefinition:
        """Get skill by name.

        Args:
            name: Skill name.

        Returns:
            Skill definition.

        Raises:
            KeyError: If skill not found.
        """
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not found")
        return self._skills[name]

    def has(self, name: str) -> bool:
        """Check if skill exists.

        Args:
            name: Skill name.

        Returns:
            True if skill exists.
        """
        return name in self._skills

    def list_names(self) -> list[str]:
        """List available skill names.

        Returns:
            List of skill names.
        """
        return list(self._skills.keys())

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get skill definitions for LLM.

        Returns:
            List of skill definitions with name, description, and input_schema.
        """
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "input_schema": skill.input_schema,
            }
            for skill in self._skills.values()
        ]

    def __len__(self) -> int:
        """Get number of registered skills."""
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        """Check if skill is registered."""
        return name in self._skills

    def __iter__(self):
        """Iterate over skill definitions."""
        return iter(self._skills.values())
