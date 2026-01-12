"""Tools for invoking skills."""

from typing import Any

from ash.skills import SkillContext, SkillExecutor, SkillRegistry
from ash.tools.base import Tool, ToolContext, ToolResult


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
        )


class WriteSkillTool(Tool):
    """Create new skills with quality guidance."""

    def __init__(self, executor: SkillExecutor) -> None:
        """Initialize tool.

        Args:
            executor: Skill executor.
        """
        self._executor = executor

    @property
    def name(self) -> str:
        return "write_skill"

    @property
    def description(self) -> str:
        return (
            "Create a new skill. The skill will be saved to the workspace "
            "and can be invoked with use_skill. If the skill needs an API, "
            "it will use web_search to find the correct endpoint."
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

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Create a skill.

        Args:
            input_data: Contains 'goal' and optional 'name'.
            context: Execution context.

        Returns:
            Skill creation result.
        """
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

        return ToolResult.success(
            result.content,
            iterations=result.iterations,
        )
