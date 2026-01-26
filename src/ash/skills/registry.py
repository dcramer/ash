"""Skill registry for discovering and loading skills from multiple sources.

Loading precedence (later sources override earlier):
1. Bundled - Built-in skills (lowest priority)
2. Installed - Externally installed from repos/local paths
3. User - User skills (~/.ash/skills/)
4. Workspace - Project-specific skills (highest priority)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ash.config.paths import get_user_skills_path
from ash.skills.base import SkillDefinition, SkillSourceType

if TYPE_CHECKING:
    from ash.config.models import SkillConfig  # noqa: F401

logger = logging.getLogger(__name__)

FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


class SkillRegistry:
    """Registry for skill definitions loaded from multiple sources.

    Skills are loaded in order of precedence:
    1. Bundled (lowest) - built-in skills
    2. Installed - from [[skills.sources]] in config
    3. User - ~/.ash/skills/
    4. Workspace (highest) - workspace/skills/
    """

    def __init__(self, skill_config: dict[str, SkillConfig] | None = None) -> None:
        self._skills: dict[str, SkillDefinition] = {}
        self._skill_sources: dict[str, Path] = {}
        self._skill_config = skill_config or {}

    def discover(
        self,
        workspace_path: Path,
        *,
        include_bundled: bool = True,
        include_installed: bool = True,
        include_user: bool = True,
    ) -> None:
        """Discover skills from all sources in precedence order.

        Args:
            workspace_path: Path to workspace (workspace/skills/ for project skills)
            include_bundled: Load bundled skills (lowest priority)
            include_installed: Load installed skills from ~/.ash/skills.installed/
            include_user: Load user skills from ~/.ash/skills/
        """
        # 1. Bundled skills (lowest priority)
        if include_bundled:
            self._load_bundled_skills()

        # 2. Installed skills (from external sources)
        if include_installed:
            self._load_installed_skills()

        # 3. User skills (~/.ash/skills/)
        if include_user:
            user_skills_dir = get_user_skills_path()
            if user_skills_dir.exists():
                self._load_from_directory(
                    user_skills_dir,
                    source_type=SkillSourceType.USER,
                )

        # 4. Workspace skills (highest priority)
        skills_dir = workspace_path / "skills"
        if skills_dir.exists():
            self._load_from_directory(
                skills_dir,
                source_type=SkillSourceType.WORKSPACE,
            )
        else:
            logger.debug(f"Workspace skills directory not found: {skills_dir}")

    def _load_bundled_skills(self) -> None:
        """Load built-in skills from the package."""
        bundled_dir = Path(__file__).parent / "bundled"
        if bundled_dir.exists():
            self._load_from_directory(
                bundled_dir,
                source_type=SkillSourceType.BUNDLED,
            )

    def _load_installed_skills(self) -> None:
        """Load skills from installed sources (repos and local paths)."""
        from ash.skills.installer import SkillInstaller

        installer = SkillInstaller()
        installed_dirs = installer.get_installed_skills_dirs()

        for skills_dir in installed_dirs:
            # Get source info from installer metadata
            source_repo = None
            source_ref = None

            for source in installer.list_installed():
                install_path = Path(source.install_path)
                # Check if this directory is from this source
                if skills_dir == install_path or skills_dir == install_path / "skills":
                    source_repo = source.repo
                    source_ref = source.ref
                    break

            self._load_from_directory(
                skills_dir,
                source_type=SkillSourceType.INSTALLED,
                source_repo=source_repo,
                source_ref=source_ref,
            )

    def _load_from_directory(
        self,
        skills_dir: Path,
        source_type: SkillSourceType = SkillSourceType.WORKSPACE,
        source_repo: str | None = None,
        source_ref: str | None = None,
    ) -> None:
        """Load skills from a directory.

        Args:
            skills_dir: Directory containing skills
            source_type: Type of source (bundled, installed, user, workspace)
            source_repo: GitHub repo (owner/repo) if installed from repo
            source_ref: Git ref if installed from repo
        """
        if not skills_dir.exists():
            return

        count_before = len(self._skills)

        for skill_dir in skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    try:
                        self._load_markdown_skill(
                            skill_file,
                            default_name=skill_dir.name,
                            source_type=source_type,
                            source_repo=source_repo,
                            source_ref=source_ref,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load skill from {skill_file}: {e}")

        for md_file in skills_dir.glob("*.md"):
            try:
                self._load_markdown_skill(
                    md_file,
                    source_type=source_type,
                    source_repo=source_repo,
                    source_ref=source_ref,
                )
            except Exception as e:
                logger.warning(f"Failed to load skill from {md_file}: {e}")

        for pattern in ("*.yaml", "*.yml"):
            for yaml_file in skills_dir.glob(pattern):
                try:
                    self._load_yaml_skill(
                        yaml_file,
                        source_type=source_type,
                        source_repo=source_repo,
                        source_ref=source_ref,
                    )
                except Exception as e:
                    logger.warning(f"Failed to load skill from {yaml_file}: {e}")

        count_loaded = len(self._skills) - count_before
        if count_loaded > 0:
            logger.info(f"Loaded {count_loaded} skills from {skills_dir}")

    def _create_skill(
        self,
        name: str,
        data: dict[str, Any],
        instructions: str,
        skill_path: Path | None,
        source_type: SkillSourceType = SkillSourceType.WORKSPACE,
        source_repo: str | None = None,
        source_ref: str | None = None,
    ) -> SkillDefinition:
        return SkillDefinition(
            name=name,
            description=data["description"],
            instructions=instructions,
            skill_path=skill_path,
            authors=data.get("authors", []),
            rationale=data.get("rationale"),
            opt_in=data.get("opt_in", False),
            source_type=source_type,
            source_repo=source_repo,
            source_ref=source_ref,
            env=data.get("env", []),
            packages=data.get("packages", []),
            tools=data.get("tools", []),
            model=data.get("model"),
            max_iterations=data.get("max_iterations", 10),
        )

    def _should_include_skill(self, skill: SkillDefinition) -> bool:
        """Check if a skill should be included based on config.

        Returns False for:
        - Opt-in skills without explicit enabled = true in config
        - Any skill with enabled = false in config

        Returns True otherwise.
        """
        config = self._skill_config.get(skill.name)

        # Check if explicitly disabled
        if config is not None and not config.enabled:
            logger.debug(f"Skill '{skill.name}' disabled in config")
            return False

        # Opt-in skills require explicit enablement
        if skill.opt_in:
            if config is None or not config.enabled:
                logger.debug(
                    f"Opt-in skill '{skill.name}' not enabled "
                    f"(add [skills.{skill.name}] enabled = true)"
                )
                return False

        return True

    def _register_skill(self, skill: SkillDefinition, source_path: Path) -> None:
        # Check if skill should be included based on opt-in/enabled settings
        if not self._should_include_skill(skill):
            return

        if skill.name in self._skills:
            existing_source = self._skill_sources.get(skill.name)
            if existing_source and existing_source != source_path:
                logger.debug(
                    f"Skill '{skill.name}' from {existing_source} "
                    f"overridden by {source_path}"
                )

        self._skills[skill.name] = skill
        self._skill_sources[skill.name] = source_path
        logger.debug(f"Loaded skill: {skill.name} from {source_path}")

    def _load_markdown_skill(
        self,
        path: Path,
        default_name: str | None = None,
        source_type: SkillSourceType = SkillSourceType.WORKSPACE,
        source_repo: str | None = None,
        source_ref: str | None = None,
    ) -> None:
        content = path.read_text()

        match = FRONTMATTER_PATTERN.match(content)
        if not match:
            raise ValueError("No YAML frontmatter found (must start with ---)")

        data = yaml.safe_load(match.group(1))
        if not isinstance(data, dict):
            raise ValueError("Frontmatter must be a YAML mapping")

        instructions = content[match.end() :].strip()
        if not instructions:
            raise ValueError("Skill missing instructions (markdown body)")

        if "description" not in data:
            raise ValueError("Skill missing required field: description")

        name = data.get("name") or default_name or path.stem
        skill_path = path.parent if path.name == "SKILL.md" else None

        skill = self._create_skill(
            name, data, instructions, skill_path, source_type, source_repo, source_ref
        )
        self._register_skill(skill, path)

    def _load_yaml_skill(
        self,
        path: Path,
        source_type: SkillSourceType = SkillSourceType.WORKSPACE,
        source_repo: str | None = None,
        source_ref: str | None = None,
    ) -> None:
        with path.open() as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid skill file: expected dict, got {type(data)}")

        if "description" not in data:
            raise ValueError("Skill missing required field: description")
        if "instructions" not in data:
            raise ValueError("Skill missing required field: instructions")

        name = data.get("name", path.stem)
        skill = self._create_skill(
            name, data, data["instructions"], None, source_type, source_repo, source_ref
        )
        self._register_skill(skill, path)

    def register(self, skill: SkillDefinition) -> None:
        self._skills[skill.name] = skill
        logger.debug(f"Registered skill: {skill.name}")

    def get(self, name: str) -> SkillDefinition:
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not found")
        return self._skills[name]

    def has(self, name: str) -> bool:
        return name in self._skills

    def list_names(self) -> list[str]:
        return list(self._skills.keys())

    def list_available(self) -> list[SkillDefinition]:
        return list(self._skills.values())

    def reload_workspace(self, workspace_path: Path) -> int:
        """Reload workspace skills only (preserves other sources)."""
        count_before = len(self._skills)
        skills_dir = workspace_path / "skills"
        if skills_dir.exists():
            self._load_from_directory(
                skills_dir,
                source_type=SkillSourceType.WORKSPACE,
            )
        return len(self._skills) - count_before

    def validate_skill_file(self, path: Path) -> tuple[bool, str | None]:
        if not path.exists():
            return False, f"File not found: {path}"

        if path.suffix != ".md":
            return False, f"Expected .md file, got: {path.name}"

        try:
            content = path.read_text()
        except Exception as e:
            return False, f"Failed to read file: {e}"

        match = FRONTMATTER_PATTERN.match(content)
        if not match:
            return False, "No YAML frontmatter found (must start with ---)"

        try:
            data = yaml.safe_load(match.group(1))
        except yaml.YAMLError as e:
            return False, f"Invalid YAML in frontmatter: {e}"

        if not isinstance(data, dict):
            return False, "Frontmatter must be a YAML mapping"

        if "description" not in data:
            return False, "Missing required field: description"

        if not content[match.end() :].strip():
            return False, "Missing instructions (markdown body after frontmatter)"

        return True, None

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __iter__(self):
        return iter(self._skills.values())
