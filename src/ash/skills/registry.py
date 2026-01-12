"""Skill registry for discovering and loading skills from workspace."""

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from ash.skills.base import SkillDefinition, SkillRequirements

logger = logging.getLogger(__name__)

# Regex to match YAML frontmatter: starts with ---, ends with ---
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

# Path to bundled skills (relative to this file)
BUNDLED_SKILLS_DIR = Path(__file__).parent / "bundled"


def _resolve_env_refs(value: str) -> str:
    """Resolve $VAR references in a value.

    Args:
        value: Value that may contain $VAR references.

    Returns:
        Value with $VAR references resolved from environment.
    """
    if not value.startswith("$"):
        return value

    env_var = value[1:]
    return os.environ.get(env_var, "")


class SkillRegistry:
    """Registry for skill definitions.

    Loads skills from:
    1. Bundled skills (shipped with Ash)
    2. Workspace skills (can override bundled)

    Supports markdown files with YAML frontmatter (preferred) or pure YAML.
    """

    def __init__(
        self,
        central_config: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """Initialize empty registry.

        Args:
            central_config: Central skill config from ~/.ash/config.toml.
                           Dict mapping skill name to config values.
                           e.g., {"check-muni": {"API_KEY": "abc123"}}
        """
        self._skills: dict[str, SkillDefinition] = {}
        self._central_config = central_config or {}

    def load_bundled(self) -> None:
        """Load bundled skills shipped with Ash.

        Bundled skills are loaded first, workspace skills can override them.
        """
        if not BUNDLED_SKILLS_DIR.exists():
            logger.debug("No bundled skills directory found")
            return

        self._load_from_directory(BUNDLED_SKILLS_DIR, source="bundled")

    def load_dynamic_skills(self) -> None:
        """Load built-in dynamic skills.

        Dynamic skills are registered programmatically rather than from SKILL.md files.
        They build their SubagentConfig at runtime with injected context.
        """
        # Import and register each dynamic skill module
        from ash.skills import research, write_skill

        research.register(self)
        write_skill.register(self)

        logger.debug("Loaded dynamic skills: research, write-skill")

    def discover(self, workspace_path: Path, *, include_bundled: bool = True) -> None:
        """Load skills from bundled, dynamic, and workspace directories.

        Load order:
        1. Bundled skills (from SKILL.md files)
        2. Dynamic skills (programmatic, self-registering)
        3. Workspace skills (can override any of the above)

        Supports:
        - Directory format: skills/<name>/SKILL.md (preferred)
        - Flat markdown: skills/<name>.md (convenience)
        - Pure YAML: skills/<name>.yaml (backward compatibility)

        Args:
            workspace_path: Path to workspace directory.
            include_bundled: Whether to load bundled skills (default True).
        """
        # Load bundled skills first
        if include_bundled:
            self.load_bundled()
            self.load_dynamic_skills()

        # Then load workspace skills (can override bundled)
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
        )

    def _load_skill_config(self, skill_path: Path) -> dict[str, str]:
        """Load config.toml from skill directory.

        Args:
            skill_path: Path to skill directory.

        Returns:
            Dict of config values (may be empty).
        """
        config_file = skill_path / "config.toml"
        if not config_file.exists():
            return {}

        try:
            import tomllib

            with config_file.open("rb") as f:
                data = tomllib.load(f)

            # Flatten to string values and resolve env refs
            config = {}
            for key, value in data.items():
                if isinstance(value, str):
                    config[key] = _resolve_env_refs(value)
                else:
                    config[key] = str(value)
            return config
        except Exception as e:
            logger.warning(f"Failed to load skill config from {config_file}: {e}")
            return {}

    def _resolve_config_values(
        self,
        skill_name: str,
        config_spec: list[str],
        skill_path: Path | None,
    ) -> dict[str, str]:
        """Resolve config values from layered sources.

        Resolution order (first match wins):
        1. Skill-local config.toml
        2. Central config from ~/.ash/config.toml
        3. Environment variables
        4. Defaults from config spec

        Args:
            skill_name: Name of the skill.
            config_spec: List of config specs (e.g., ["API_KEY", "TIMEOUT=30"]).
            skill_path: Path to skill directory (for loading config.toml).

        Returns:
            Dict of resolved config values.
        """
        resolved = {}

        # Parse defaults from spec using shared method
        defaults = {}
        names = []
        for item in config_spec:
            name, default = SkillDefinition.parse_config_spec(item)
            names.append(name)
            if default is not None:
                defaults[name] = default

        # Load skill-local config
        skill_config = {}
        if skill_path and skill_path.is_dir():
            skill_config = self._load_skill_config(skill_path)

        # Get central config for this skill
        central_config = self._central_config.get(skill_name, {})

        # Resolve each config value
        for name in names:
            # 1. Skill-local config.toml
            if name in skill_config:
                resolved[name] = skill_config[name]
            # 2. Central config
            elif name in central_config:
                value = central_config[name]
                resolved[name] = (
                    _resolve_env_refs(value) if isinstance(value, str) else str(value)
                )
            # 3. Environment variable
            elif os.environ.get(name):
                resolved[name] = os.environ[name]
            # 4. Default from spec
            elif name in defaults:
                resolved[name] = defaults[name]

        return resolved

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
            skill_path: Path to skill directory (for config.toml loading).

        Returns:
            SkillDefinition instance.
        """
        requirements = self._parse_requirements(data)

        # Parse config spec
        config_spec = data.get("config", [])
        if not isinstance(config_spec, list):
            config_spec = []

        # Resolve config values
        config_values = self._resolve_config_values(name, config_spec, skill_path)

        # Parse subagent flag (supports both 'subagent' and legacy 'execution_mode')
        subagent = data.get("subagent", False)
        if not subagent and data.get("execution_mode") == "subagent":
            subagent = True  # backward compat

        return SkillDefinition(
            name=name,
            description=description,
            instructions=instructions,
            subagent=subagent,
            model=data.get("model") or data.get("preferred_model"),  # backward compat
            required_tools=data.get("required_tools", []),
            input_schema=data.get("input_schema", {}),
            max_iterations=data.get("max_iterations", 5),
            requires=requirements,
            config=config_spec,
            config_values=config_values,
            skill_path=skill_path,
        )

    def _register_skill(self, skill: SkillDefinition, source_path: Path) -> None:
        """Register a skill, logging warnings for overrides.

        Args:
            skill: Skill definition to register.
            source_path: Path where skill was loaded from.
        """
        if skill.name in self._skills:
            logger.warning(f"Skill '{skill.name}' overwritten by {source_path}")

        # Check availability and log if not available
        is_available, reason = skill.is_available()
        if not is_available:
            logger.debug(f"Skill '{skill.name}' not available: {reason}")

        self._skills[skill.name] = skill
        logger.debug(f"Loaded skill: {skill.name} from {source_path}")

    def _load_markdown_skill(self, path: Path, default_name: str | None = None) -> None:
        """Load a skill from a markdown file with YAML frontmatter.

        Format:
            ---
            description: What the skill does
            model: fast  # optional model alias
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

        # YAML files don't have a skill directory, so no skill-local config
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

    def register_dynamic(
        self,
        name: str,
        description: str,
        build_config: Any,  # Callable that returns SubagentConfig
        required_tools: list[str] | None = None,
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        """Register a dynamic skill that builds its config at runtime.

        Dynamic skills use a build_config callable instead of static instructions.
        They always execute as subagents.

        Args:
            name: Skill name.
            description: One-line description.
            build_config: Callable that takes (input_data, **kwargs) and returns SubagentConfig.
            required_tools: Tools the skill needs (for availability checking).
            input_schema: JSON Schema for skill inputs.
        """
        skill = SkillDefinition(
            name=name,
            description=description,
            instructions="",  # Dynamic skills don't use static instructions
            subagent=True,  # Dynamic skills are always subagents
            required_tools=required_tools or [],
            input_schema=input_schema or {},
            build_config=build_config,
        )
        self._skills[skill.name] = skill
        logger.debug(f"Registered dynamic skill: {skill.name}")

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

    def __len__(self) -> int:
        """Get number of registered skills."""
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        """Check if skill is registered."""
        return name in self._skills

    def __iter__(self):
        """Iterate over available skill definitions."""
        return iter(self.list_available())
