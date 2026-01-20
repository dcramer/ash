"""Skill registry for discovering and loading skills from workspace."""

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from ash.skills.base import SkillDefinition

logger = logging.getLogger(__name__)

FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


class SkillRegistry:
    """Registry for skill definitions loaded from workspace/skills/."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}
        self._skill_sources: dict[str, Path] = {}

    def discover(self, workspace_path: Path, *, include_bundled: bool = True) -> None:
        skills_dir = workspace_path / "skills"
        if not skills_dir.exists():
            logger.debug(f"Workspace skills directory not found: {skills_dir}")
            return

        self._load_from_directory(skills_dir)

    def _load_from_directory(self, skills_dir: Path) -> None:
        if not skills_dir.exists():
            return

        count_before = len(self._skills)

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

        for md_file in skills_dir.glob("*.md"):
            try:
                self._load_markdown_skill(md_file)
            except Exception as e:
                logger.warning(f"Failed to load skill from {md_file}: {e}")

        for pattern in ("*.yaml", "*.yml"):
            for yaml_file in skills_dir.glob(pattern):
                try:
                    self._load_yaml_skill(yaml_file)
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
    ) -> SkillDefinition:
        return SkillDefinition(
            name=name,
            description=data["description"],
            instructions=instructions,
            skill_path=skill_path,
            authors=data.get("authors", []),
            rationale=data.get("rationale"),
            env=data.get("env", []),
            packages=data.get("packages", []),
            allowed_tools=data.get("allowed_tools", []),
            model=data.get("model"),
            max_iterations=data.get("max_iterations", 10),
        )

    def _register_skill(self, skill: SkillDefinition, source_path: Path) -> None:
        if skill.name in self._skills:
            existing_source = self._skill_sources.get(skill.name)
            if existing_source and existing_source != source_path:
                logger.warning(f"Skill '{skill.name}' overwritten by {source_path}")

        self._skills[skill.name] = skill
        self._skill_sources[skill.name] = source_path
        logger.debug(f"Loaded skill: {skill.name} from {source_path}")

    def _load_markdown_skill(self, path: Path, default_name: str | None = None) -> None:
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

        skill = self._create_skill(name, data, instructions, skill_path)
        self._register_skill(skill, path)

    def _load_yaml_skill(self, path: Path) -> None:
        with path.open() as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid skill file: expected dict, got {type(data)}")

        if "description" not in data:
            raise ValueError("Skill missing required field: description")
        if "instructions" not in data:
            raise ValueError("Skill missing required field: instructions")

        name = data.get("name", path.stem)
        skill = self._create_skill(name, data, data["instructions"], None)
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
        count_before = len(self._skills)
        skills_dir = workspace_path / "skills"
        if skills_dir.exists():
            self._load_from_directory(skills_dir)
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
