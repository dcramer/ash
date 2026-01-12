"""Skill execution with sub-agent loop."""

import json
import logging
import time
from typing import Any

from ash.config.models import AshConfig, ConfigError
from ash.llm import ToolDefinition
from ash.llm.registry import create_llm_provider
from ash.llm.types import ContentBlock, Message, Role
from ash.llm.types import ToolResult as LLMToolResult
from ash.skills.base import SkillContext, SkillDefinition, SkillResult, SubagentConfig
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

    async def _run_subagent(
        self,
        config: SubagentConfig,
        context: SkillContext,
        name: str = "subagent",
    ) -> SkillResult:
        """Run a subagent with the given configuration.

        This is the core subagent loop used by skills and tools that spawn
        isolated LLM conversations with restricted tool access.

        Args:
            config: Subagent configuration.
            context: Skill execution context.
            name: Name for logging (e.g., skill name or "write-skill").

        Returns:
            Skill result with subagent output.
        """
        start_time = time.monotonic()

        # Resolve model
        model_alias = config.model or "default"
        try:
            model_config = self._config.get_model(model_alias)
        except ConfigError:
            logger.warning(f"Model alias '{model_alias}' not found, using default")
            model_config = self._config.default_model

        api_key = self._config.resolve_api_key(
            model_alias if model_alias in self._config.models else "default"
        )
        provider = create_llm_provider(
            model_config.provider,
            api_key=api_key.get_secret_value() if api_key else None,
        )

        # Build tool definitions - filter to allowed tools if specified
        all_tool_defs = self._tool_executor.get_definitions()
        if config.allowed_tools:
            allowed_set = set(config.allowed_tools)
            tool_definitions = [
                ToolDefinition(
                    name=td["name"],
                    description=td["description"],
                    input_schema=td["input_schema"],
                )
                for td in all_tool_defs
                if td["name"] in allowed_set
            ]
        else:
            tool_definitions = [
                ToolDefinition(
                    name=td["name"],
                    description=td["description"],
                    input_schema=td["input_schema"],
                )
                for td in all_tool_defs
            ]

        # Initialize conversation
        messages: list[Message] = [
            Message(role=Role.USER, content=config.initial_message)
        ]

        iterations = 0
        result_text = ""

        logger.info(
            f"Starting {name} (model={model_config.model}, "
            f"max_iterations={config.max_iterations})"
        )

        # Subagent loop
        while iterations < config.max_iterations:
            iterations += 1
            logger.debug(f"{name} iteration {iterations}/{config.max_iterations}")

            try:
                response = await provider.complete(
                    messages=messages,
                    model=model_config.model,
                    tools=tool_definitions if tool_definitions else None,
                    system=config.system_prompt,
                    max_tokens=model_config.max_tokens,
                    temperature=model_config.temperature,
                )
            except Exception as e:
                logger.exception(f"{name} LLM call failed")
                return SkillResult.error(f"LLM call failed: {e}")

            # Add assistant message to conversation
            messages.append(response.message)

            # Check for tool uses
            tool_uses = response.message.get_tool_uses()
            if not tool_uses:
                # No tool calls, we're done
                result_text = response.message.get_text() or ""
                break

            # Execute tools
            tool_context = ToolContext(
                session_id=context.session_id,
                user_id=context.user_id,
                chat_id=context.chat_id,
                env=config.env,
            )

            tool_results: list[ContentBlock] = []
            for tool_use in tool_uses:
                logger.debug(f"{name} executing tool: {tool_use.name}")

                result = await self._tool_executor.execute(
                    tool_use.name,
                    tool_use.input,
                    tool_context,
                )

                tool_results.append(
                    LLMToolResult(
                        tool_use_id=tool_use.id,
                        content=result.content,
                        is_error=result.is_error,
                    )
                )

            # Add tool results to conversation
            messages.append(Message(role=Role.USER, content=tool_results))

        # Log execution
        duration_ms = int((time.monotonic() - start_time) * 1000)
        logger.info(f"{name} completed in {duration_ms}ms ({iterations} iterations)")

        # Check if we hit max iterations
        if iterations >= config.max_iterations and not result_text:
            result_text = (
                f"Reached maximum iterations ({config.max_iterations}). "
                "Result may be incomplete."
            )
            return SkillResult(
                content=result_text,
                is_error=False,
                iterations=iterations,
            )

        return SkillResult.success(result_text, iterations=iterations)

    def has_skill(self, skill_name: str) -> bool:
        """Check if a skill exists.

        Args:
            skill_name: Name of the skill.

        Returns:
            True if skill exists.
        """
        return self._registry.has(skill_name)

    async def execute(
        self,
        skill_name: str,
        input_data: dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """Execute a skill.

        Routes to inline, subagent, or dynamic execution based on skill type.

        Args:
            skill_name: Name of skill to execute.
            input_data: Input data for skill.
            context: Skill execution context.

        Returns:
            Skill execution result.
        """
        # Get skill from registry
        try:
            skill = self._registry.get(skill_name)
        except KeyError:
            return SkillResult.error(f"Skill '{skill_name}' not found")

        # Check availability
        is_available, reason = skill.is_available()
        if not is_available:
            return SkillResult.error(f"Skill '{skill_name}' not available: {reason}")

        # Validate required tools are available
        error = self._validate_tools(skill)
        if error:
            return SkillResult.error(error)

        # Validate input
        error = self._validate_input(skill, input_data)
        if error:
            return SkillResult.error(f"Invalid input: {error}")

        # Route based on skill type
        if skill.is_dynamic:
            return await self._execute_dynamic(skill, input_data, context)
        elif skill.subagent:
            return await self._execute_subagent(skill, input_data, context)
        else:
            return await self._execute_inline(skill, input_data, context)

    async def _execute_inline(
        self,
        skill: SkillDefinition,
        input_data: dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """Execute skill in inline mode.

        Returns skill instructions for the main agent to follow using its tools.
        No sub-agent loop is created.

        Args:
            skill: Skill definition.
            input_data: Input data for skill.
            context: Skill execution context.

        Returns:
            Skill result containing instructions for main agent.
        """
        logger.info(f"Executing skill '{skill.name}' in inline mode")

        # Build instructions with {baseDir} substitution
        instructions = skill.instructions
        if skill.skill_path:
            instructions = instructions.replace("{baseDir}", str(skill.skill_path))

        # Append input data if provided
        if input_data:
            instructions += (
                f"\n\n## Input\n```json\n{json.dumps(input_data, indent=2)}\n```"
            )

        # Return instructions for main agent to follow
        return SkillResult.success(
            f"## Skill: {skill.name}\n\n{instructions}",
            iterations=0,
        )

    async def _execute_subagent(
        self,
        skill: SkillDefinition,
        input_data: dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """Execute skill in subagent mode with isolated sub-agent loop.

        Args:
            skill: Skill definition.
            input_data: Input data for skill.
            context: Skill execution context.

        Returns:
            Skill execution result.
        """
        # Build system prompt with input data
        system_prompt = self._build_system_prompt(skill, input_data)

        # Build SKILL_* env vars from skill config
        skill_env = {
            f"SKILL_{name.upper()}": value
            for name, value in skill.config_values.items()
        }

        # Resolve model alias (per-skill config > skill.model > default)
        skill_config = self._config.skills.get(skill.name, {})
        model_alias = skill_config.get("model") or skill.model

        # Build subagent config
        config = SubagentConfig(
            system_prompt=system_prompt,
            allowed_tools=skill.required_tools,
            max_iterations=skill.max_iterations,
            model=model_alias,
            env=skill_env,
        )

        return await self._run_subagent(config, context, name=f"skill:{skill.name}")

    async def _execute_dynamic(
        self,
        skill: SkillDefinition,
        input_data: dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """Execute a dynamic skill that builds its config at runtime.

        Dynamic skills use a build_config callable to create their SubagentConfig,
        allowing them to inject runtime context like available tools.

        Args:
            skill: Dynamic skill definition (must have build_config set).
            input_data: Input data for skill.
            context: Skill execution context.

        Returns:
            Skill execution result.
        """
        if not skill.build_config:
            return SkillResult.error(
                f"Skill '{skill.name}' is marked as dynamic but has no build_config"
            )

        # Build config with standard context kwargs
        try:
            config = skill.build_config(
                input_data,
                tool_definitions=self._tool_executor.get_definitions(),
                workspace_path=self._config.workspace,
            )
        except Exception as e:
            logger.exception(f"Failed to build config for dynamic skill '{skill.name}'")
            return SkillResult.error(f"Failed to build skill config: {e}")

        return await self._run_subagent(config, context, name=skill.name)
