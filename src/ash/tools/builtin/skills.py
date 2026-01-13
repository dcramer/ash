"""Tools for invoking skills."""

import logging
from pathlib import Path
from typing import Any

from ash.skills import SkillContext, SkillExecutor, SkillRegistry
from ash.tools.base import Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class UseSkillTool(Tool):
    """Invoke a skill by name."""

    def __init__(
        self,
        registry: SkillRegistry,
        executor: SkillExecutor,
    ) -> None:
        """Initialize tool.

        Args:
            registry: Skill registry.
            executor: Skill executor.
        """
        self._registry = registry
        self._executor = executor

    @property
    def name(self) -> str:
        return "use_skill"

    @property
    def description(self) -> str:
        return (
            "Invoke a skill by name. Skills are reusable behaviors "
            "that orchestrate tools with specific instructions. "
            "Available skills are listed in the system prompt."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "Name of the skill to invoke.",
                },
                "input": {
                    "type": "object",
                    "description": "Input parameters for the skill.",
                    "default": {},
                },
            },
            "required": ["skill"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Invoke a skill.

        Args:
            input_data: Must contain 'skill' key.
            context: Execution context.

        Returns:
            Skill execution result.
        """
        skill_name = input_data.get("skill")
        if not skill_name:
            return ToolResult.error("Missing required parameter: skill")

        skill_input = input_data.get("input", {})

        # Build skill context from tool context
        skill_context = SkillContext(
            session_id=context.session_id,
            user_id=context.user_id,
            chat_id=context.chat_id,
            input_data=skill_input,
        )

        # Execute skill
        result = await self._executor.execute(
            skill_name,
            skill_input,
            skill_context,
        )

        if result.is_error:
            return ToolResult.error(result.content)

        return ToolResult.success(
            result.content,
            iterations=result.iterations,
            skill_env=result.skill_env,
        )


class WriteSkillTool(Tool):
    """Create or update skills with quality guidance."""

    def __init__(
        self,
        executor: SkillExecutor,
        registry: SkillRegistry,
        workspace_path: Path | None = None,
    ) -> None:
        """Initialize tool.

        Args:
            executor: Skill executor.
            registry: Skill registry (for reloading after skill creation).
            workspace_path: Workspace path (for reloading skills).
        """
        self._executor = executor
        self._registry = registry
        self._workspace_path = workspace_path

    @property
    def name(self) -> str:
        return "write_skill"

    @property
    def description(self) -> str:
        return (
            "Create or update a skill. If a skill with the given name already exists, "
            "it will be updated. The skill will be saved to the workspace "
            "and can be invoked with use_skill."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Name for the skill (lowercase, hyphens). "
                        "Required if user specifies a name."
                    ),
                },
                "goal": {
                    "type": "string",
                    "description": "What the skill should accomplish.",
                },
            },
            "required": ["goal"],
        }

    def _read_existing_skill(self, skill_name: str) -> str | None:
        """Read existing skill content if it exists.

        Args:
            skill_name: Name of the skill to read.

        Returns:
            Skill content if exists, None otherwise.
        """
        if not self._workspace_path or not skill_name:
            return None

        skill_path = self._workspace_path / "skills" / skill_name / "SKILL.md"
        if skill_path.exists():
            try:
                return skill_path.read_text()
            except Exception as e:
                logger.warning(f"Failed to read existing skill at {skill_path}: {e}")
                return None
        return None

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Create or update a skill.

        Args:
            input_data: Contains 'goal' and optional 'name'.
            context: Execution context.

        Returns:
            Skill creation result.
        """
        skill_name = input_data.get("name")

        # Check for existing skill to update
        existing_skill = self._read_existing_skill(skill_name) if skill_name else None
        if existing_skill:
            # Add to input_data for write-skill to use
            input_data = {**input_data, "existing_skill": existing_skill}

        skill_context = SkillContext(
            session_id=context.session_id,
            user_id=context.user_id,
            chat_id=context.chat_id,
            input_data=input_data,
        )

        result = await self._executor.execute(
            "write-skill",
            input_data,
            skill_context,
        )

        if result.is_error:
            return ToolResult.error(result.content)

        # Validate skill creation
        if self._workspace_path:
            skills_dir = self._workspace_path / "skills"
            new_count = self._registry.reload_workspace(self._workspace_path)

            if new_count > 0:
                reload_msg = f"\n\n[Loaded {new_count} new skill(s) - available immediately via use_skill]"
                return ToolResult.success(
                    result.content + reload_msg,
                    iterations=result.iterations,
                )

            # No new skills found - check for validation errors
            # Look for skill files in expected locations
            validation_errors: list[str] = []
            checked_paths: list[Path] = []

            if skills_dir.exists():
                # Check specific paths if we know the skill name
                if skill_name:
                    checked_paths.append(skills_dir / skill_name / "SKILL.md")
                    checked_paths.append(skills_dir / f"{skill_name}.md")

                # Also scan for any .md files that might be new skills
                for skill_dir in skills_dir.iterdir():
                    if skill_dir.is_dir():
                        skill_file = skill_dir / "SKILL.md"
                        if skill_file.exists() and skill_file not in checked_paths:
                            checked_paths.append(skill_file)

            # Validate each path we find
            for path in checked_paths:
                if path.exists():
                    is_valid, error = self._registry.validate_skill_file(path)
                    if not is_valid:
                        validation_errors.append(f"  - {path.name}: {error}")

            if validation_errors:
                error_detail = "\n".join(validation_errors)
                return ToolResult.error(
                    f"{result.content}\n\n"
                    f"Skill file was created but has validation errors:\n{error_detail}\n\n"
                    f"The skill must have YAML frontmatter with a `description` field "
                    f"and markdown instructions in the body."
                )

            # File not found at expected locations
            warning_msg = (
                "\n\n**Warning:** No new skill was discovered after creation. "
                "The skill file may have been written to the wrong location.\n\n"
                "Expected: /workspace/skills/<name>/SKILL.md\n"
                "Check the output above to see where the file was written."
            )
            return ToolResult.success(
                result.content + warning_msg,
                iterations=result.iterations,
            )

        return ToolResult.success(
            result.content,
            iterations=result.iterations,
        )
