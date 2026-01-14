"""Skill invocation tool."""

import logging
from typing import TYPE_CHECKING, Any

from ash.agents.base import Agent, AgentConfig, AgentContext
from ash.skills.base import SkillDefinition
from ash.tools.base import Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from ash.agents import AgentExecutor
    from ash.config import AshConfig
    from ash.skills import SkillRegistry

logger = logging.getLogger(__name__)

# Wrapper guidance prepended to all skill system prompts
SKILL_AGENT_WRAPPER = """You are a skill executor. Your job is to run the skill instructions below and report results.

## How to Execute

1. Follow the instructions in the skill definition
2. Run any commands or tools specified
3. Report what happened - include actual output

## Handling Errors

When a command fails or returns an error:
- Report the error message to the user
- STOP - do not attempt to fix, debug, or work around the problem
- The user will decide whether to invoke the skill-writer to fix the skill

**NEVER do any of the following when something fails:**
- Read the script source to understand why it failed
- Copy or modify script files
- Use sed, awk, or other tools to edit files
- Write inline scripts to diagnose the issue
- Try alternative approaches not in the instructions

If the skill is broken, say so and stop. That's useful information.

## Output

Your response goes back to the main agent, who relays it to the user.
- Include actual command output, not just summaries
- If something failed, include the error message
- Be concise - the user wants results, not a narrative

---

"""


class SkillAgent(Agent):
    """Ephemeral agent wrapper for a skill definition.

    Converts a SkillDefinition into an Agent so it can be executed
    via AgentExecutor with the standard agent loop.
    """

    def __init__(
        self,
        skill: SkillDefinition,
        model_override: str | None = None,
    ) -> None:
        """Initialize skill agent.

        Args:
            skill: Skill definition to wrap.
            model_override: Optional model alias to override skill's default.
        """
        self._skill = skill
        self._model_override = model_override

    @property
    def config(self) -> AgentConfig:
        """Return agent configuration derived from skill."""
        return AgentConfig(
            name=f"skill:{self._skill.name}",
            description=self._skill.description,
            system_prompt=self._skill.instructions,
            allowed_tools=self._skill.allowed_tools,
            max_iterations=self._skill.max_iterations,
            model=self._model_override or self._skill.model,
            is_skill_agent=True,
        )

    def build_system_prompt(self, context: AgentContext) -> str:
        """Build system prompt with wrapper guidance and optional context injection.

        Args:
            context: Execution context with optional user-provided context.

        Returns:
            System prompt string with wrapper + skill instructions + context.
        """
        # Start with wrapper guidance + skill instructions
        prompt = SKILL_AGENT_WRAPPER + self._skill.instructions

        # Inject user-provided context if available
        user_context = context.input_data.get("context", "")
        if user_context:
            prompt += f"\n\n## Context\n\n{user_context}"

        return prompt


class UseSkillTool(Tool):
    """Invoke a skill with isolated execution.

    Skills run as subagents with their own LLM loops, tool restrictions,
    and scoped environments (API keys injected from config).
    """

    def __init__(
        self,
        registry: "SkillRegistry",
        executor: "AgentExecutor",
        config: "AshConfig",
    ) -> None:
        """Initialize the tool.

        Args:
            registry: Skill registry to look up skills.
            executor: Agent executor to run skill agents.
            config: Application configuration for skill settings.
        """
        self._registry = registry
        self._executor = executor
        self._config = config

    @property
    def name(self) -> str:
        """Tool name."""
        return "use_skill"

    @property
    def description(self) -> str:
        """Tool description."""
        skills = self._registry.list_available()
        if not skills:
            return "Invoke a skill (none available)"
        skill_list = ", ".join(s.name for s in skills)
        return f"Invoke a skill with isolated execution. Available: {skill_list}"

    @property
    def input_schema(self) -> dict[str, Any]:
        """Input schema for the tool."""
        return {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "Name of the skill to invoke",
                },
                "message": {
                    "type": "string",
                    "description": "Task/message for the skill to work on",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context to help the skill understand the task",
                },
            },
            "required": ["skill", "message"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        """Execute the tool.

        Args:
            input_data: Tool input with skill name, message, and optional context.
            context: Optional tool execution context.

        Returns:
            ToolResult with skill output.
        """
        skill_name = input_data.get("skill")
        message = input_data.get("message")
        user_context = input_data.get("context", "")

        if not skill_name:
            return ToolResult.error("Missing required field: skill")

        if not message:
            return ToolResult.error("Missing required field: message")

        # Look up skill
        if not self._registry.has(skill_name):
            available = ", ".join(self._registry.list_names())
            return ToolResult.error(
                f"Skill '{skill_name}' not found. Available: {available}"
            )

        skill = self._registry.get(skill_name)

        # Check if skill is enabled in config
        skill_config = self._config.skills.get(skill_name)
        if skill_config and not skill_config.enabled:
            return ToolResult.error(f"Skill '{skill_name}' is disabled in config")

        # Check if skill has required env vars that aren't configured
        if skill.env:
            config_env = skill_config.get_env_vars() if skill_config else {}
            missing = [var for var in skill.env if var not in config_env]
            if missing:
                return ToolResult.error(
                    f"Skill '{skill_name}' requires configuration.\n\n"
                    f"Add to ~/.ash/config.toml:\n\n"
                    f"[skills.{skill_name}]\n"
                    + "\n".join(f'{var} = "your-value-here"' for var in missing)
                )

        # Build scoped environment from config
        env = self._build_skill_environment(skill, skill_config)

        # Determine model override from config
        model_override = skill_config.model if skill_config else None

        # Create ephemeral agent from skill
        agent = SkillAgent(skill, model_override=model_override)

        # Build agent context with user-provided context
        agent_context = AgentContext(
            session_id=context.session_id if context else None,
            user_id=context.user_id if context else None,
            chat_id=context.chat_id if context else None,
            input_data={"context": user_context},
        )

        logger.info(f"Invoking skill '{skill_name}' with message: {message[:100]}...")

        # Execute with scoped environment
        result = await self._executor.execute(
            agent, message, agent_context, environment=env
        )

        if result.is_error:
            return ToolResult.error(result.content)

        return ToolResult.success(
            result.content, iterations=result.iterations, skill=skill_name
        )

    def _build_skill_environment(
        self,
        skill: SkillDefinition,
        skill_config: Any | None,
    ) -> dict[str, str]:
        """Build scoped environment for skill execution.

        Args:
            skill: Skill definition with env requirements.
            skill_config: Optional config for the skill.

        Returns:
            Dict of env var name to value.
        """
        env: dict[str, str] = {}

        if not skill_config:
            # No config for this skill - warn if skill needs env vars
            if skill.env:
                logger.warning(
                    f"Skill '{skill.name}' needs env vars {skill.env} "
                    f"but no [skills.{skill.name}] config section found"
                )
            return env

        # Get env vars from config
        config_env = skill_config.get_env_vars()

        # Only inject env vars that the skill declared it needs
        for var_name in skill.env:
            if var_name in config_env:
                env[var_name] = config_env[var_name]
            else:
                logger.warning(
                    f"Skill '{skill.name}' needs {var_name} but not found in "
                    f"[skills.{skill.name}] config"
                )

        return env
