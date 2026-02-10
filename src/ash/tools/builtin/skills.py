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

# Built-in skills that are handled specially (not loaded from SKILL.md files)
BUILTIN_SKILLS = {
    "claude-code": (
        "Delegate tasks to Claude Code CLI with full permissions. "
        "Each invocation is a fresh conversation. To continue a previous task, "
        "have Claude Code write output to a file, then reference it in the next "
        "prompt using @filename syntax."
    ),
}

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

Your final response is the skill's output - this is how results reach the user.

- Include actual command output, not just summaries
- If something failed, include the error message
- Be concise - the user wants results, not a narrative

For long-running tasks, use `send_message` for progress updates only (e.g., "Processing file 3 of 10...").
Never use `send_message` for the final result - that must be in your response.

---

"""


def format_skill_result(content: str, skill_name: str) -> str:
    """Format skill result with structured tags for LLM clarity."""
    from ash.tools.base import format_subagent_result

    return format_subagent_result(content, "skill", skill_name)


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
            tools=self._skill.tools,
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
        # Start with wrapper guidance
        prompt = SKILL_AGENT_WRAPPER

        # Add provenance if available
        if self._skill.authors or self._skill.rationale:
            prompt += f"## Skill: {self._skill.name}\n"
            if self._skill.authors:
                prompt += f"**Authors:** {', '.join(self._skill.authors)}\n"
            if self._skill.rationale:
                prompt += f"**Rationale:** {self._skill.rationale}\n"
            prompt += "\n"

        # Add skill instructions
        prompt += self._skill.instructions

        # Inject user-provided context if available
        user_context = context.input_data.get("context", "")
        if user_context:
            prompt += f"\n\n## Context\n\n{user_context}"

        # Add voice guidance for user-facing messages
        if context.voice:
            prompt += f"""

## Communication Style (for user-facing messages only)

{context.voice}

IMPORTANT: Apply this style ONLY to interrupt() prompts that users will see.
Do NOT apply it to tool outputs, file content, or technical results."""

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
        voice: str | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            registry: Skill registry to look up skills.
            executor: Agent executor to run skill agents.
            config: Application configuration for skill settings.
            voice: Optional communication style for user-facing skill messages.
        """
        self._registry = registry
        self._executor = executor
        self._config = config
        self._voice = voice

    @property
    def name(self) -> str:
        return "use_skill"

    @property
    def description(self) -> str:
        # Combine registry skills with built-in skills
        skill_names = [s.name for s in self._registry.list_available()]
        skill_names.extend(BUILTIN_SKILLS.keys())
        if not skill_names:
            return "Invoke a skill (none available)"
        skill_list = ", ".join(sorted(set(skill_names)))
        return f"Invoke a skill with isolated execution. Available: {skill_list}"

    @property
    def input_schema(self) -> dict[str, Any]:
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

    def _build_skill_environment(
        self,
        skill: SkillDefinition,
        skill_config: Any,
    ) -> dict[str, str]:
        """Build environment dict for skill execution."""
        if not skill_config:
            return {}
        config_env = skill_config.get_env_vars()
        env: dict[str, str] = {}
        for var_name in skill.env:
            if var_name in config_env:
                env[var_name] = config_env[var_name]
            else:
                logger.warning(
                    f"Skill '{skill.name}' needs {var_name} but not found in "
                    f"[skills.{skill.name}] config"
                )
        return env

    async def _execute_claude_code(
        self,
        message: str,
        context: ToolContext | None,
    ) -> ToolResult:
        """Execute the claude-code built-in skill.

        This skill delegates to the Claude Code CLI with full permissions.
        Model is read from [models.claude-code].model config.
        """
        from ash.agents.builtin.claude_code import ClaudeCodeAgent

        skill_config = self._config.skills.get("claude-code")

        if skill_config and not skill_config.enabled:
            return ToolResult.error("Skill 'claude-code' is disabled in config")

        # Get model from [models.claude-code] if configured, default to opus
        if "claude-code" in self._config.models:
            model = self._config.models["claude-code"].model
        else:
            model = "opus"

        agent = ClaudeCodeAgent()

        if context:
            agent_context = AgentContext.from_tool_context(context)
        else:
            agent_context = AgentContext()

        logger.info(f"Invoking claude-code skill with message: {message[:100]}...")

        result = await agent.execute_passthrough(message, agent_context, model=model)

        if result.is_error:
            return ToolResult.error(result.content)

        return ToolResult.success(
            format_skill_result(result.content, "claude-code"),
            skill="claude-code",
        )

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        skill_name = input_data.get("skill")
        message = input_data.get("message")
        user_context = input_data.get("context", "")

        if not skill_name:
            return ToolResult.error("Missing required field: skill")

        if not message:
            return ToolResult.error("Missing required field: message")

        # Handle built-in skills specially
        if skill_name == "claude-code":
            return await self._execute_claude_code(message, context)

        if not self._registry.has(skill_name):
            self._registry.reload_workspace(self._config.workspace)
            if not self._registry.has(skill_name):
                # Include built-in skills in available list
                available = set(self._registry.list_names())
                available.update(BUILTIN_SKILLS.keys())
                return ToolResult.error(
                    f"Skill '{skill_name}' not found. Available: {', '.join(sorted(available))}"
                )

        skill = self._registry.get(skill_name)
        skill_config = self._config.skills.get(skill_name)

        if skill_config and not skill_config.enabled:
            return ToolResult.error(f"Skill '{skill_name}' is disabled in config")

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

        env = self._build_skill_environment(skill, skill_config)
        model_override = skill_config.model if skill_config else None
        agent = SkillAgent(skill, model_override=model_override)

        if context:
            agent_context = AgentContext.from_tool_context(
                context, input_data={"context": user_context}, voice=self._voice
            )
        else:
            agent_context = AgentContext(
                input_data={"context": user_context}, voice=self._voice
            )

        logger.info(f"Invoking skill '{skill_name}' with message: {message[:100]}...")

        # Get session info from context for subagent logging
        session_manager, tool_use_id = (
            context.get_session_info() if context else (None, None)
        )

        result = await self._executor.execute(
            agent,
            message,
            agent_context,
            environment=env,
            session_manager=session_manager,
            parent_tool_use_id=tool_use_id,
        )

        if result.is_error:
            return ToolResult.error(result.content)

        return ToolResult.success(
            format_skill_result(result.content, skill_name),
            iterations=result.iterations,
            skill=skill_name,
        )
