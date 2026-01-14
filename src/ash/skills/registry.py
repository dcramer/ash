"""Skill registry for discovering and loading skills from workspace."""

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from ash.skills.base import SkillDefinition, SkillRequirements

logger = logging.getLogger(__name__)

# Regex to match YAML frontmatter: starts with ---, ends with ---
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


class SkillRegistry:
    """Registry for skill definitions.

    Loads skills from workspace/skills/ directory only.
    Skills are markdown files with YAML frontmatter that the agent reads and follows.

    Supports markdown files with YAML frontmatter (preferred) or pure YAML.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._skills: dict[str, SkillDefinition] = {}
        self._skill_sources: dict[str, Path] = {}

    def load_bundled(self) -> None:
        """No-op for backward compatibility.

        Bundled skills have been removed. Use Agents for built-in capabilities.
        """
        pass

    def discover(self, workspace_path: Path, *, include_bundled: bool = True) -> None:
        """Load skills from workspace directory.

        Supports:
        - Directory format: skills/<name>/SKILL.md (preferred)
        - Flat markdown: skills/<name>.md (convenience)
        - Pure YAML: skills/<name>.yaml (backward compatibility)

        Args:
            workspace_path: Path to workspace directory.
            include_bundled: Ignored (kept for backward compatibility).
        """
        skills_dir = workspace_path / "skills"
        if not skills_dir.exists():
            logger.debug(f"Workspace skills directory not found: {skills_dir}")
            return

        self._load_from_directory(skills_dir, source="workspace")

    def _load_from_directory(self, skills_dir: Path, source: str = "unknown") -> None:
        """Load skills from a directory.

        Args:
            skills_dir: Path to skills directory.
            source: Source label for logging (bundled, workspace).
        """
        if not skills_dir.exists():
            return

        count_before = len(self._skills)

        # Preferred: skills/<name>/SKILL.md
        for skill_dir in skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    try:
                        self._load_markdown_skill(
                            skill_file, default_name=skill_dir.name
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load skill from {skill_file}: {e}")

        # Also support flat markdown files
        for md_file in skills_dir.glob("*.md"):
            try:
                self._load_markdown_skill(md_file)
            except Exception as e:
                logger.warning(f"Failed to load skill from {md_file}: {e}")

        # Also support pure YAML for backward compatibility
        for pattern in ("*.yaml", "*.yml"):
            for yaml_file in skills_dir.glob(pattern):
                try:
                    self._load_yaml_skill(yaml_file)
                except Exception as e:
                    logger.warning(f"Failed to load skill from {yaml_file}: {e}")

        count_loaded = len(self._skills) - count_before
        if count_loaded > 0:
            logger.info(f"Loaded {count_loaded} skills from {source} ({skills_dir})")

    def _parse_requirements(self, data: dict[str, Any]) -> SkillRequirements:
        """Parse requirements from skill data.

        Args:
            data: Skill data dict (from YAML).

        Returns:
            SkillRequirements instance.
        """
        requires = data.get("requires", {})
        if not isinstance(requires, dict):
            return SkillRequirements()

        return SkillRequirements(
            bins=requires.get("bins", []),
            env=requires.get("env", []),
            os=requires.get("os", []),
            apt_packages=requires.get("apt_packages", []),
            python_packages=requires.get("python_packages", []),
            python_tools=requires.get("python_tools", []),
        )

    def _create_skill_definition(
        self,
        name: str,
        description: str,
        instructions: str,
        data: dict[str, Any],
        skill_path: Path | None,
    ) -> SkillDefinition:
        """Create a SkillDefinition from parsed data.

        Args:
            name: Skill name.
            description: Skill description.
            instructions: Skill instructions (markdown body or YAML field).
            data: Full parsed data dict (for optional fields).
            skill_path: Path to skill directory.

        Returns:
            SkillDefinition instance.
        """
        requirements = self._parse_requirements(data)

        return SkillDefinition(
            name=name,
            description=description,
            instructions=instructions,
            required_tools=data.get("required_tools", []),
            input_schema=data.get("input_schema", {}),
            requires=requirements,
            skill_path=skill_path,
        )

    def _register_skill(self, skill: SkillDefinition, source_path: Path) -> None:
        """Register a skill, logging warnings for overrides.

        Args:
            skill: Skill definition to register.
            source_path: Path where skill was loaded from.
        """
        if skill.name in self._skills:
            existing_source = self._skill_sources.get(skill.name)
            if existing_source and existing_source != source_path:
                logger.warning(f"Skill '{skill.name}' overwritten by {source_path}")

        # Check availability and log if not available
        is_available, reason = skill.is_available()
        if not is_available:
            logger.debug(f"Skill '{skill.name}' not available: {reason}")

        self._skills[skill.name] = skill
        self._skill_sources[skill.name] = source_path
        logger.debug(f"Loaded skill: {skill.name} from {source_path}")

    def _load_markdown_skill(self, path: Path, default_name: str | None = None) -> None:
        """Load a skill from a markdown file with YAML frontmatter.

        Format:
            ---
            description: What the skill does
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

        # Determine skill path (directory containing SKILL.md)
        skill_path = path.parent if path.name == "SKILL.md" else None

        skill = self._create_skill_definition(
            name=name,
            description=data["description"],
            instructions=instructions,
            data=data,
            skill_path=skill_path,
        )
        self._register_skill(skill, path)

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

        skill = self._create_skill_definition(
            name=name,
            description=data["description"],
            instructions=data["instructions"],
            data=data,
            skill_path=None,
        )
        self._register_skill(skill, path)

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
        """List all registered skill names (including unavailable).

        Returns:
            List of skill names.
        """
        return list(self._skills.keys())

    def list_available(self) -> list[SkillDefinition]:
        """List skills available on the current system.

        Returns:
            List of available skill definitions.
        """
        return [skill for skill in self._skills.values() if skill.is_available()[0]]

    def get_definitions(
        self, include_unavailable: bool = False
    ) -> list[dict[str, Any]]:
        """Get skill definitions for LLM.

        Args:
            include_unavailable: If True, include skills that don't meet requirements.

        Returns:
            List of skill definitions with name, description, and input_schema.
        """
        skills = self._skills.values() if include_unavailable else self.list_available()
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "input_schema": skill.input_schema,
            }
            for skill in skills
        ]

    def reload_workspace(self, workspace_path: Path) -> int:
        """Reload skills from workspace directory.

        This allows discovering newly created skills without restarting.
        Only loads from workspace, doesn't reload bundled skills.

        Args:
            workspace_path: Path to workspace directory.

        Returns:
            Number of new skills loaded.
        """
        count_before = len(self._skills)
        skills_dir = workspace_path / "skills"
        if skills_dir.exists():
            self._load_from_directory(skills_dir, source="workspace")
        return len(self._skills) - count_before

    def validate_skill_file(self, path: Path) -> tuple[bool, str | None]:
        """Validate a skill file without loading it into the registry.

        Checks that the file:
        - Exists
        - Has valid YAML frontmatter
        - Contains required 'description' field
        - Has instructions in the markdown body

        Args:
            path: Path to skill file (SKILL.md or <name>.md).

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        if not path.exists():
            return False, f"File not found: {path}"

        if path.suffix != ".md":
            return False, f"Expected .md file, got: {path.name}"

        try:
            content = path.read_text()
        except Exception as e:
            return False, f"Failed to read file: {e}"

        # Check frontmatter
        match = FRONTMATTER_PATTERN.match(content)
        if not match:
            return False, "No YAML frontmatter found (must start with ---)"

        try:
            frontmatter_yaml = match.group(1)
            data = yaml.safe_load(frontmatter_yaml)
        except yaml.YAMLError as e:
            return False, f"Invalid YAML in frontmatter: {e}"

        if not isinstance(data, dict):
            return False, "Frontmatter must be a YAML mapping"

        if "description" not in data:
            return False, "Missing required field: description"

        instructions = content[match.end() :].strip()
        if not instructions:
            return False, "Missing instructions (markdown body after frontmatter)"

        return True, None

    def __len__(self) -> int:
        """Get number of registered skills."""
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        """Check if skill is registered."""
        return name in self._skills

    def __iter__(self):
        """Iterate over available skill definitions."""
        return iter(self.list_available())
