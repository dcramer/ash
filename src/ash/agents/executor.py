"""Agent executor for running isolated subagent loops."""

import logging
from typing import TYPE_CHECKING

from ash.agents.base import Agent, AgentContext, AgentResult
from ash.core.session import SessionState
from ash.llm.types import Role, ToolDefinition
from ash.tools.base import ToolContext

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.llm.base import LLMProvider
    from ash.tools import ToolExecutor

logger = logging.getLogger(__name__)


class AgentExecutor:
    """Execute agents in isolated subagent loops."""

    def __init__(
        self,
        llm_provider: "LLMProvider",
        tool_executor: "ToolExecutor",
        config: "AshConfig",
    ) -> None:
        self._llm = llm_provider
        self._tools = tool_executor
        self._config = config

    def _get_tool_definitions(
        self, allowed_tools: list[str], is_skill_agent: bool
    ) -> list[ToolDefinition]:
        all_defs = self._tools.get_definitions()

        if allowed_tools:
            defs = [d for d in all_defs if d.name in allowed_tools]
        else:
            defs = all_defs

        if is_skill_agent:
            defs = [t for t in defs if t.name != "use_skill"]

        return defs

    async def execute(
        self,
        agent: Agent,
        input_message: str,
        context: AgentContext,
        environment: dict[str, str] | None = None,
    ) -> AgentResult:
        agent_config = agent.config
        logger.info(
            f"Executing agent '{agent_config.name}' with input: {input_message[:100]}..."
        )

        overrides = self._config.agents.get(agent_config.name)
        model_alias = (
            overrides.model if overrides and overrides.model else agent_config.model
        )
        max_iterations = (
            overrides.max_iterations
            if overrides and overrides.max_iterations
            else agent_config.max_iterations
        )

        resolved_model: str | None = None
        if model_alias:
            try:
                resolved_model = self._config.get_model(model_alias).model
            except Exception as e:
                available = ", ".join(sorted(self._config.models.keys()))
                logger.error(
                    f"Agent '{agent_config.name}' has invalid model alias '{model_alias}'. "
                    f"Available aliases: {available}"
                )
                return AgentResult.error(f"Invalid model alias: {model_alias}. {e}")

        logger.info(
            f"Agent '{agent_config.name}' using model: {resolved_model or 'default'}"
        )

        system_prompt = agent.build_system_prompt(context)
        tool_definitions = self._get_tool_definitions(
            agent_config.allowed_tools, agent_config.is_skill_agent
        )

        session = SessionState(
            session_id=f"agent-{agent_config.name}-{context.session_id or 'unknown'}",
            provider=self._config.default_model.provider,
            chat_id=context.chat_id or "",
            user_id=context.user_id or "",
        )
        session.add_user_message(input_message)

        tool_context = ToolContext(
            session_id=context.session_id,
            user_id=context.user_id,
            chat_id=context.chat_id,
            env=environment or {},
        )

        for iteration in range(1, max_iterations + 1):
            logger.debug(
                f"Agent '{agent_config.name}' iteration {iteration}/{max_iterations}"
            )

            try:
                response = await self._llm.complete(
                    messages=session.get_messages_for_llm(),
                    model=resolved_model,
                    system=system_prompt,
                    tools=tool_definitions or None,
                    max_tokens=4096,
                )
            except Exception as e:
                logger.error(f"Agent '{agent_config.name}' LLM error: {e}")
                return AgentResult.error(f"LLM error: {e}")

            message = response.message
            session.add_assistant_message(message.content)

            tool_uses = message.get_tool_uses()
            if not tool_uses:
                text = message.get_text()
                logger.info(
                    f"Agent '{agent_config.name}' completed in {iteration} iterations"
                )
                return AgentResult.success(text, iterations=iteration)

            for tool_use in tool_uses:
                if (
                    agent_config.allowed_tools
                    and tool_use.name not in agent_config.allowed_tools
                ):
                    session.add_tool_result(
                        tool_use.id,
                        f"Tool '{tool_use.name}' is not available to this agent",
                        is_error=True,
                    )
                    continue

                try:
                    result = await self._tools.execute(
                        tool_use.name,
                        tool_use.input,
                        context=tool_context,
                    )
                    session.add_tool_result(
                        tool_use.id,
                        result.content,
                        is_error=result.is_error,
                    )
                except Exception as e:
                    logger.error(f"Agent tool execution error: {e}")
                    session.add_tool_result(
                        tool_use.id,
                        f"Tool error: {e}",
                        is_error=True,
                    )

        logger.warning(
            f"Agent '{agent_config.name}' hit max iterations ({max_iterations})"
        )

        last_text = ""
        if session.messages:
            last_msg = session.messages[-1]
            if last_msg.role == Role.ASSISTANT:
                last_text = last_msg.get_text()

        return AgentResult(
            content=last_text or f"Agent reached iteration limit ({max_iterations})",
            is_error=True,
            iterations=max_iterations,
        )
