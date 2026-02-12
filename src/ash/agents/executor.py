"""Agent executor for running isolated subagent loops."""

import logging
import uuid
from typing import TYPE_CHECKING

from ash.agents.base import Agent
from ash.agents.types import AgentContext, AgentResult, CheckpointState
from ash.core.session import SessionState
from ash.llm.types import Role, ToolDefinition
from ash.tools.base import ToolContext

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.llm.base import LLMProvider
    from ash.sessions.manager import SessionManager
    from ash.tools import ToolExecutor

logger = logging.getLogger(__name__)

# Keywords that indicate user wants to cancel rather than continue
CANCEL_KEYWORDS = {"cancel", "abort", "nevermind", "never mind", "stop", "quit"}


def is_cancel_message(message: str) -> bool:
    """Check if a message indicates cancellation intent."""
    return message.lower().strip() in CANCEL_KEYWORDS


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
        self,
        tools: list[str],
        is_skill_agent: bool,
        supports_checkpointing: bool,
    ) -> list[ToolDefinition]:
        all_defs = self._tools.get_definitions()

        # Build set of excluded tools
        excluded = set()
        if is_skill_agent:
            excluded.add("use_skill")
        if not supports_checkpointing:
            excluded.add("interrupt")

        # Filter by tools whitelist and exclusions in a single pass
        if tools:
            tools_set = set(tools)
            return [
                d for d in all_defs if d.name in tools_set and d.name not in excluded
            ]

        return [d for d in all_defs if d.name not in excluded]

    async def _log_assistant_message(
        self,
        session_manager: "SessionManager",
        agent_session_id: str,
        content: str | list,
        iteration: int,
    ) -> None:
        """Log assistant message and any tool uses to the session.

        Args:
            session_manager: Session manager for logging.
            agent_session_id: The subagent session ID.
            content: The message content blocks.
            iteration: Current iteration number.
        """
        from ash.sessions.utils import content_block_to_dict

        # Handle string content (simple text response)
        if isinstance(content, str):
            await session_manager.add_assistant_message(
                content=content,
                metadata={"iteration": iteration},
                agent_session_id=agent_session_id,
            )
            return

        # Convert content blocks to serializable format and log
        # add_assistant_message handles tool use extraction automatically
        serialized = [content_block_to_dict(b) for b in content]
        await session_manager.add_assistant_message(
            content=serialized,
            metadata={"iteration": iteration},
            agent_session_id=agent_session_id,
        )

    async def execute(
        self,
        agent: Agent,
        input_message: str,
        context: AgentContext,
        environment: dict[str, str] | None = None,
        resume_from: CheckpointState | None = None,
        user_response: str | None = None,
        session_manager: "SessionManager | None" = None,
        parent_tool_use_id: str | None = None,
    ) -> AgentResult:
        """Execute an agent.

        Args:
            agent: The agent to execute.
            input_message: Initial message/task for the agent.
            context: Execution context.
            environment: Optional environment variables for tools.
            resume_from: Optional checkpoint to resume from.
            user_response: User's response when resuming from checkpoint.
            session_manager: Optional session manager for logging subagent activity.
            parent_tool_use_id: Tool use ID that invoked this subagent (for logging).

        Returns:
            AgentResult with content, or interrupted result with checkpoint.
        """
        from ash.logging import log_context

        with log_context(chat_id=context.chat_id, session_id=context.session_id):
            return await self._execute_inner(
                agent=agent,
                input_message=input_message,
                context=context,
                environment=environment,
                resume_from=resume_from,
                user_response=user_response,
                session_manager=session_manager,
                parent_tool_use_id=parent_tool_use_id,
            )

    async def _execute_inner(
        self,
        agent: Agent,
        input_message: str,
        context: AgentContext,
        environment: dict[str, str] | None = None,
        resume_from: CheckpointState | None = None,
        user_response: str | None = None,
        session_manager: "SessionManager | None" = None,
        parent_tool_use_id: str | None = None,
    ) -> AgentResult:
        """Inner implementation of execute (runs with log context)."""
        agent_config = agent.config
        agent_session_id: str | None = None

        # Start subagent session logging if session_manager is provided
        if session_manager and parent_tool_use_id:
            agent_type = "skill" if agent_config.is_skill_agent else "agent"
            agent_session_id = await session_manager.start_agent_session(
                parent_tool_use_id=parent_tool_use_id,
                agent_type=agent_type,
                agent_name=agent_config.name,
            )
            logger.debug(
                f"Started agent session {agent_session_id} for {agent_type} "
                f"'{agent_config.name}'"
            )

        # Handle passthrough agents - they bypass the LLM loop entirely
        if agent_config.is_passthrough:
            logger.info(
                f"Executing passthrough agent '{agent_config.name}' with input: "
                f"{input_message[:100]}..."
            )
            return await agent.execute_passthrough(input_message, context)

        # Handle resume from checkpoint
        if resume_from is not None:
            if user_response is None:
                return AgentResult.error("user_response required when resuming")

            if resume_from.is_expired():
                return AgentResult.error("Checkpoint has expired. Please start over.")

            # Validate checkpoint belongs to this agent
            if resume_from.agent_name != agent_config.name:
                return AgentResult.error(
                    f"Checkpoint belongs to '{resume_from.agent_name}', "
                    f"not '{agent_config.name}'"
                )

            logger.info(
                f"Resuming agent '{agent_config.name}' from checkpoint "
                f"{resume_from.checkpoint_id}"
            )

            # Restore session from checkpoint
            try:
                session = SessionState.from_json(resume_from.session_json)
            except Exception as e:
                logger.error(f"Failed to restore checkpoint session: {e}")
                return AgentResult.error(f"Checkpoint session corrupted: {e}")
            start_iteration = resume_from.iteration

            # Inject user response as the tool result for the interrupt call
            session.add_tool_result(
                tool_use_id=resume_from.tool_use_id,
                content=user_response,
                is_error=False,
            )
        else:
            logger.info(
                f"Executing agent '{agent_config.name}' with input: "
                f"{input_message[:100]}..."
            )
            start_iteration = 1
            session = SessionState(
                session_id=f"agent-{agent_config.name}-{context.session_id or 'unknown'}",
                provider=self._config.default_model.provider,
                chat_id=context.chat_id or "",
                user_id=context.user_id or "",
            )
            session.add_user_message(input_message)

            # Log the input message to session
            if session_manager and agent_session_id:
                await session_manager.add_user_message(
                    content=input_message,
                    agent_session_id=agent_session_id,
                )

        overrides = self._config.agents.get(agent_config.name)
        model_alias = (overrides.model if overrides else None) or agent_config.model
        max_iterations = (
            overrides.max_iterations if overrides else None
        ) or agent_config.max_iterations

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
            agent_config.tools,
            agent_config.is_skill_agent,
            agent_config.supports_checkpointing,
        )

        tool_context = ToolContext.from_agent_context(context, env=environment)

        for iteration in range(start_iteration, max_iterations + 1):
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

            # Log assistant message to session
            if session_manager and agent_session_id:
                await self._log_assistant_message(
                    session_manager, agent_session_id, message.content, iteration
                )

            tool_uses = message.get_tool_uses()
            if not tool_uses:
                text = message.get_text()
                logger.info(
                    f"Agent '{agent_config.name}' completed in {iteration} iterations"
                )
                return AgentResult.success(text, iterations=iteration)

            # Check for interrupt tool first - it takes priority over other tools
            interrupt_tool = next((t for t in tool_uses if t.name == "interrupt"), None)
            if interrupt_tool:
                # Add error results for any other tools that were called
                for tool_use in tool_uses:
                    if tool_use.name != "interrupt":
                        session.add_tool_result(
                            tool_use.id,
                            "Skipped: agent interrupted for user input",
                            is_error=True,
                        )

                prompt = interrupt_tool.input.get("prompt", "Checkpoint reached")
                options = interrupt_tool.input.get("options")

                checkpoint = CheckpointState(
                    checkpoint_id=str(uuid.uuid4()),
                    agent_name=agent_config.name,
                    session_json=session.to_json(),
                    iteration=iteration,
                    prompt=prompt,
                    options=options,
                    tool_use_id=interrupt_tool.id,
                )

                logger.info(
                    f"Agent '{agent_config.name}' interrupted at iteration "
                    f"{iteration}: {prompt[:100]}..."
                )

                return AgentResult.interrupted(checkpoint, iterations=iteration)

            for tool_use in tool_uses:
                # Prevent agents from invoking themselves via use_agent
                if tool_use.name == "use_agent":
                    target_agent = tool_use.input.get("agent", "")
                    if target_agent == agent_config.name:
                        session.add_tool_result(
                            tool_use.id,
                            f"Agent '{agent_config.name}' cannot invoke itself",
                            is_error=True,
                        )
                        continue

                if agent_config.tools and tool_use.name not in agent_config.tools:
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
                    output = result.content
                    is_error = result.is_error
                except Exception as e:
                    logger.error(f"Agent tool execution error: {e}")
                    output = f"Tool error: {e}"
                    is_error = True

                session.add_tool_result(tool_use.id, output, is_error=is_error)
                if session_manager and agent_session_id:
                    await session_manager.add_tool_result(
                        tool_use_id=tool_use.id,
                        output=output,
                        success=not is_error,
                        agent_session_id=agent_session_id,
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
            content=last_text
            or "The agent couldn't complete within the allowed steps. It may have made partial progress.",
            is_error=True,
            iterations=max_iterations,
        )
