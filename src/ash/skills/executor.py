"""Skill execution with sub-agent loop."""

import json
import logging
import time
from typing import Any

from ash.config.models import AshConfig, ConfigError
from ash.llm import LLMProvider, ToolDefinition
from ash.llm.registry import create_llm_provider
from ash.llm.types import Message, Role, TextContent, ToolUse
from ash.skills.base import SkillContext, SkillDefinition, SkillResult
from ash.skills.registry import SkillRegistry
from ash.tools.base import ToolContext
from ash.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


class SkillExecutor:
    """Execute skills with sub-agent loop."""

    def __init__(
        self,
        registry: SkillRegistry,
        tool_executor: ToolExecutor,
        config: AshConfig,
    ) -> None:
        """Initialize skill executor.

        Args:
            registry: Skill registry.
            tool_executor: Tool executor for running tools.
            config: Application config for model resolution.
        """
        self._registry = registry
        self._tool_executor = tool_executor
        self._config = config

    def _resolve_model(
        self, skill: SkillDefinition
    ) -> tuple[LLMProvider, str, float | None, int]:
        """Resolve model alias to provider and model config.

        Args:
            skill: Skill definition with preferred_model.

        Returns:
            Tuple of (provider, model, temperature, max_tokens).
        """
        alias = skill.preferred_model or "default"

        try:
            model_config = self._config.get_model(alias)
        except ConfigError:
            logger.warning(
                f"Model alias '{alias}' not found, using default model"
            )
            model_config = self._config.default_model

        api_key = self._config.resolve_api_key(alias if alias in self._config.models else "default")
        provider = create_llm_provider(
            model_config.provider,
            api_key=api_key.get_secret_value() if api_key else None,
        )

        return (
            provider,
            model_config.model,
            model_config.temperature,
            model_config.max_tokens,
        )

    def _validate_tools(self, skill: SkillDefinition) -> str | None:
        """Validate that all required tools are available.

        Args:
            skill: Skill definition.

        Returns:
            Error message if validation fails, None otherwise.
        """
        for tool_name in skill.required_tools:
            if tool_name not in self._tool_executor.available_tools:
                return f"Skill requires tool '{tool_name}' which is not available"
        return None

    def _validate_input(
        self, skill: SkillDefinition, input_data: dict[str, Any]
    ) -> str | None:
        """Validate input against skill's input_schema.

        Args:
            skill: Skill definition.
            input_data: Input data to validate.

        Returns:
            Error message if validation fails, None otherwise.
        """
        if not skill.input_schema:
            return None

        schema = skill.input_schema
        required = schema.get("required", [])

        for field_name in required:
            if field_name not in input_data:
                return f"Missing required input field: {field_name}"

        return None

    def _get_tool_definitions(
        self, skill: SkillDefinition
    ) -> list[ToolDefinition]:
        """Get tool definitions for the skill.

        If skill has required_tools, only include those.
        Otherwise, include all available tools.

        Args:
            skill: Skill definition.

        Returns:
            List of tool definitions.
        """
        definitions = []
        tool_defs = self._tool_executor.get_definitions()

        for tool_def in tool_defs:
            if not skill.required_tools or tool_def["name"] in skill.required_tools:
                definitions.append(
                    ToolDefinition(
                        name=tool_def["name"],
                        description=tool_def["description"],
                        input_schema=tool_def["input_schema"],
                    )
                )

        return definitions

    def _build_system_prompt(
        self, skill: SkillDefinition, input_data: dict[str, Any]
    ) -> str:
        """Build system prompt for skill execution.

        Args:
            skill: Skill definition.
            input_data: Input data.

        Returns:
            System prompt.
        """
        prompt = skill.instructions

        if input_data:
            prompt += f"\n\n## Input\n```json\n{json.dumps(input_data, indent=2)}\n```"

        return prompt

    async def execute(
        self,
        skill_name: str,
        input_data: dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """Execute skill with sub-agent loop.

        Args:
            skill_name: Name of skill to execute.
            input_data: Input data for skill.
            context: Skill execution context.

        Returns:
            Skill execution result.
        """
        start_time = time.monotonic()

        # Get skill
        try:
            skill = self._registry.get(skill_name)
        except KeyError:
            return SkillResult.error(f"Skill '{skill_name}' not found")

        # Check availability
        is_available, reason = skill.is_available()
        if not is_available:
            return SkillResult.error(f"Skill '{skill_name}' not available: {reason}")

        # Validate tools
        error = self._validate_tools(skill)
        if error:
            return SkillResult.error(error)

        # Validate input
        error = self._validate_input(skill, input_data)
        if error:
            return SkillResult.error(f"Invalid input: {error}")

        # Resolve model
        provider, model, temperature, max_tokens = self._resolve_model(skill)

        # Build prompts
        system_prompt = self._build_system_prompt(skill, input_data)
        tool_definitions = self._get_tool_definitions(skill)

        # Initialize conversation
        messages: list[Message] = [
            Message(
                role=Role.USER,
                content="Execute the skill according to the instructions and input provided.",
            )
        ]

        iterations = 0
        result_text = ""

        # Sub-agent loop
        while iterations < skill.max_iterations:
            iterations += 1

            try:
                response = await provider.complete(
                    messages=messages,
                    model=model,
                    tools=tool_definitions if tool_definitions else None,
                    system=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                logger.exception(f"Skill '{skill_name}' LLM call failed")
                return SkillResult.error(f"LLM call failed: {e}")

            # Add assistant message to conversation
            messages.append(response.message)

            # Check for tool uses
            tool_uses = response.message.get_tool_uses()
            if not tool_uses:
                # No tool calls, we're done
                result_text = response.message.get_text() or ""
                break

            # Build SKILL_* env vars from skill config
            skill_env = {
                f"SKILL_{name.upper()}": value
                for name, value in skill.config_values.items()
            }

            # Execute tools
            tool_context = ToolContext(
                session_id=context.session_id,
                user_id=context.user_id,
                chat_id=context.chat_id,
                env=skill_env,
            )

            tool_results: list[TextContent | ToolUse | Any] = []
            for tool_use in tool_uses:
                logger.debug(f"Skill '{skill_name}' executing tool: {tool_use.name}")

                result = await self._tool_executor.execute(
                    tool_use.name,
                    tool_use.input,
                    tool_context,
                )

                from ash.llm.types import ToolResult as LLMToolResult

                tool_results.append(
                    LLMToolResult(
                        tool_use_id=tool_use.id,
                        content=result.content,
                        is_error=result.is_error,
                    )
                )

            # Add tool results to conversation
            messages.append(
                Message(
                    role=Role.USER,
                    content=tool_results,
                )
            )

        # Log execution
        duration_ms = int((time.monotonic() - start_time) * 1000)
        logger.info(
            f"Skill '{skill_name}' completed in {duration_ms}ms "
            f"({iterations} iterations)"
        )

        # Check if we hit max iterations
        if iterations >= skill.max_iterations and not result_text:
            result_text = (
                f"Skill execution reached maximum iterations ({skill.max_iterations}). "
                "Partial result may be incomplete."
            )
            return SkillResult(
                content=result_text,
                is_error=False,
                iterations=iterations,
            )

        return SkillResult.success(result_text, iterations=iterations)
