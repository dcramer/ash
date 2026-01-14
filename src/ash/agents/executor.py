"""Agent executor for running isolated subagent loops."""

import logging
from typing import TYPE_CHECKING

from ash.agents.base import Agent, AgentContext, AgentResult
from ash.core.session import SessionState
from ash.llm.types import Message, Role, ToolDefinition
from ash.tools.base import ToolContext

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.llm.base import LLMProvider
    from ash.tools import ToolExecutor

logger = logging.getLogger(__name__)


class AgentExecutor:
    """Execute agents in isolated subagent loops.

    Runs an agent's LLM loop separately from the main conversation,
    with its own session state and optionally restricted tools.
    """

    def __init__(
        self,
        llm_provider: "LLMProvider",
        tool_executor: "ToolExecutor",
        config: "AshConfig",
    ) -> None:
        """Initialize agent executor.

        Args:
            llm_provider: LLM provider for completions.
            tool_executor: Tool executor for tool calls.
            config: Application configuration.
        """
        self._llm = llm_provider
        self._tools = tool_executor
        self._config = config

    def _filter_tools(self, allowed_tools: list[str]) -> list[ToolDefinition]:
        """Filter tool definitions to allowed list.

        Args:
            allowed_tools: List of allowed tool names. Empty = all tools.

        Returns:
            Filtered list of tool definitions.
        """
        all_defs = self._tools.get_definitions()

        if allowed_tools:
            all_defs = [d for d in all_defs if d["name"] in allowed_tools]

        return [
            ToolDefinition(
                name=d["name"],
                description=d["description"],
                input_schema=d["input_schema"],
            )
            for d in all_defs
        ]

    async def execute(
        self,
        agent: Agent,
        input_message: str,
        context: AgentContext,
        environment: dict[str, str] | None = None,
    ) -> AgentResult:
        """Execute agent in isolated loop.

        Args:
            agent: Agent to execute.
            input_message: User message to start the agent.
            context: Execution context.
            environment: Optional env vars to pass to tools.

        Returns:
            AgentResult with content and metadata.
        """
        agent_config = agent.config
        logger.info(
            f"Executing agent '{agent_config.name}' with input: {input_message[:100]}..."
        )

        # Apply config overrides from [agents.<name>] section
        overrides = self._config.agents.get(agent_config.name)
        model_alias = (
            overrides.model if overrides and overrides.model else agent_config.model
        )
        max_iterations = (
            overrides.max_iterations
            if overrides and overrides.max_iterations
            else agent_config.max_iterations
        )

        # Resolve model alias to full model ID (None = use default)
        resolved_model: str | None = None
        if model_alias:
            try:
                model_config = self._config.get_model(model_alias)
                resolved_model = model_config.model
            except Exception as e:
                available = ", ".join(sorted(self._config.models.keys()))
                logger.error(
                    f"Agent '{agent_config.name}' has invalid model alias '{model_alias}'. "
                    f"Available aliases: {available}"
                )
                return AgentResult.error(f"Invalid model alias: {model_alias}. {e}")

        # Build system prompt (may inject context)
        system_prompt = agent.build_system_prompt(context)

        # Get filtered tool definitions
        tool_definitions = self._filter_tools(agent_config.allowed_tools)

        # Create isolated session for this agent
        session = SessionState(
            session_id=f"agent-{agent_config.name}-{context.session_id or 'unknown'}",
            provider=self._config.default_model.provider,
            chat_id=context.chat_id or "",
            user_id=context.user_id or "",
        )
        session.add_user_message(input_message)

        iterations = 0
        consecutive_failures = 0
        max_consecutive_failures = 3

        while iterations < max_iterations:
            iterations += 1
            logger.debug(
                f"Agent '{agent_config.name}' iteration {iterations}/{max_iterations}"
            )

            try:
                response = await self._llm.complete(
                    messages=session.get_messages_for_llm(),
                    model=resolved_model,
                    system=system_prompt,
                    tools=tool_definitions if tool_definitions else None,
                    max_tokens=4096,
                )
            except Exception as e:
                logger.error(f"Agent '{agent_config.name}' LLM error: {e}")
                return AgentResult.error(f"LLM error: {e}")

            message = response.message

            # Add assistant message to session
            session.add_assistant_message(message.content)

            # Check for tool uses
            tool_uses = message.get_tool_uses()

            if not tool_uses:
                # No tools = agent is done, return text response
                text = message.get_text()
                logger.info(
                    f"Agent '{agent_config.name}' completed in {iterations} iterations"
                )
                return AgentResult.success(text, iterations=iterations)

            # Build tool context with environment
            tool_context = ToolContext(
                session_id=context.session_id,
                user_id=context.user_id,
                chat_id=context.chat_id,
                env=environment or {},
            )

            # Execute tools
            all_failed = True
            for tool_use in tool_uses:
                # Check if tool is allowed
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
                    if not result.is_error:
                        all_failed = False
                except Exception as e:
                    logger.error(f"Agent tool execution error: {e}")
                    session.add_tool_result(
                        tool_use.id,
                        f"Tool error: {e}",
                        is_error=True,
                    )

            # Track consecutive failures
            if all_failed:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(
                        f"Agent '{agent_config.name}' stopped after "
                        f"{consecutive_failures} consecutive failed iterations"
                    )
                    return AgentResult(
                        content=f"Stopped: {consecutive_failures} consecutive iterations "
                        "with all tools failing",
                        is_error=True,
                        iterations=iterations,
                    )
            else:
                consecutive_failures = 0

        # Hit max iterations
        logger.warning(
            f"Agent '{agent_config.name}' hit max iterations ({max_iterations})"
        )

        # Try to get any text from the last message
        last_text = ""
        if session.messages:
            last_msg = session.messages[-1]
            if last_msg.role == Role.ASSISTANT:
                last_text = (
                    last_msg.content
                    if isinstance(last_msg.content, str)
                    else Message(
                        role=Role.ASSISTANT, content=last_msg.content
                    ).get_text()
                )

        return AgentResult(
            content=last_text or f"Agent reached iteration limit ({max_iterations})",
            is_error=True,
            iterations=iterations,
        )
