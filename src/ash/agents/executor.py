"""Agent executor for running isolated subagent loops."""

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING

from ash.agents.base import Agent
from ash.agents.types import (
    AgentContext,
    AgentResult,
    CheckpointState,
    StackFrame,
    TurnAction,
    TurnResult,
)
from ash.core.session import SessionState
from ash.llm.types import ToolDefinition
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

    @staticmethod
    def _build_result_metadata(tool_context: ToolContext) -> dict[str, str]:
        """Extract metadata to propagate from tool context to agent result."""
        metadata: dict[str, str] = {}
        if reply_id := tool_context.reply_to_message_id:
            metadata["reply_to_message_id"] = reply_id
        return metadata

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
        # complete is only used in interactive subagent mode (execute_turn),
        # not in batch mode (execute)
        excluded.add("complete")

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

        timeout = agent.config.timeout

        with log_context(chat_id=context.chat_id, session_id=context.session_id):
            try:
                return await asyncio.wait_for(
                    self._execute_inner(
                        agent=agent,
                        input_message=input_message,
                        context=context,
                        environment=environment,
                        resume_from=resume_from,
                        user_response=user_response,
                        session_manager=session_manager,
                        parent_tool_use_id=parent_tool_use_id,
                    ),
                    timeout=timeout,
                )
            except TimeoutError:
                logger.error(
                    "Agent '%s' timed out after %ds",
                    agent.config.name,
                    timeout,
                )
                return AgentResult.error(f"Agent timed out after {timeout} seconds")

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
        effective_tools = agent_config.get_effective_tools()
        tool_definitions = self._get_tool_definitions(
            effective_tools,
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
                result_metadata = self._build_result_metadata(tool_context)
                if not text and iteration > 1:
                    # Agent completed without producing text after tool execution
                    return AgentResult.error(
                        "Agent completed without producing a response"
                    )
                return AgentResult.success(
                    text, iterations=iteration, metadata=result_metadata
                )

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

                if effective_tools and tool_use.name not in effective_tools:
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

        last_text = session.get_last_text_response() or ""
        result_metadata = self._build_result_metadata(tool_context)

        content = (
            f"{last_text}\n\n[Agent hit the maximum number of steps and may not have finished.]"
            if last_text
            else "The agent couldn't complete within the allowed steps. It may have made partial progress."
        )

        return AgentResult(
            content=content,
            is_error=True,
            iterations=max_iterations,
            metadata=result_metadata,
        )

    # --- Interactive subagent support ---

    def _get_turn_tool_definitions(
        self,
        frame: StackFrame,
    ) -> list[ToolDefinition]:
        """Get tool definitions for an interactive turn.

        Like _get_tool_definitions but includes 'complete' and excludes
        tools not appropriate for interactive subagents.
        """
        all_defs = self._tools.get_definitions()

        excluded: set[str] = set()
        if frame.is_skill_agent:
            excluded.add("use_skill")
        # Interactive subagents use complete, not interrupt
        excluded.add("interrupt")

        if frame.effective_tools:
            tools_set = set(frame.effective_tools)
            # Always allow complete for interactive subagents
            tools_set.add("complete")
            return [
                d for d in all_defs if d.name in tools_set and d.name not in excluded
            ]

        return [d for d in all_defs if d.name not in excluded]

    @staticmethod
    def _get_unresolved_tool_uses(session: SessionState) -> list:
        """Find tool_uses from the most recent assistant message that lack results.

        Walks backward to find the last assistant message, then checks which
        tool_uses from it have not yet received tool_results.
        """
        from ash.llm.types import Role, ToolUse
        from ash.llm.types import ToolResult as LLMToolResult

        for i in range(len(session.messages) - 1, -1, -1):
            msg = session.messages[i]
            if msg.role == Role.ASSISTANT:
                if isinstance(msg.content, str):
                    return []
                tool_uses = [b for b in msg.content if isinstance(b, ToolUse)]
                if not tool_uses:
                    return []
                # Collect tool_result IDs from messages after this assistant message
                resolved: set[str] = set()
                for j in range(i + 1, len(session.messages)):
                    later = session.messages[j]
                    if isinstance(later.content, list):
                        for block in later.content:
                            if isinstance(block, LLMToolResult):
                                resolved.add(block.tool_use_id)
                return [tu for tu in tool_uses if tu.id not in resolved]
        return []

    async def execute_turn(
        self,
        frame: StackFrame,
        user_message: str | None = None,
        tool_result: tuple[str, str, bool] | None = None,
        session_manager: "SessionManager | None" = None,
    ) -> TurnResult:
        """Run one logical turn for a stack frame.

        Entry points:
        - user_message set: add user message, call LLM
        - tool_result set: inject tool_result (child completed), resume
        - Both None: first turn for a newly pushed child (session already has initial message)

        Args:
            frame: The stack frame to execute.
            user_message: Optional user message to inject.
            tool_result: Optional (tool_use_id, content, is_error) from completed child.
            session_manager: Optional session manager for logging to context.jsonl.

        Returns:
            TurnResult indicating what happened.
        """
        from ash.agents.types import ChildActivated

        session = frame.session
        agent_session_id = frame.agent_session_id
        tool_defs = self._get_turn_tool_definitions(frame)
        tool_context = ToolContext.from_agent_context(
            frame.context,
            env=frame.environment or {},
            session_manager=session_manager,
        )

        if user_message is not None:
            session.add_user_message(user_message)
            if session_manager and agent_session_id:
                await session_manager.add_user_message(
                    content=user_message,
                    agent_session_id=agent_session_id,
                )
        elif tool_result is not None:
            tu_id, content, is_error = tool_result
            session.add_tool_result(tu_id, content, is_error)
            if session_manager and agent_session_id:
                await session_manager.add_tool_result(
                    tool_use_id=tu_id,
                    output=content,
                    success=not is_error,
                    agent_session_id=agent_session_id,
                )

        while frame.iteration < frame.max_iterations:
            frame.iteration += 1

            # Check for unresolved tool_uses from a previous assistant message
            unresolved = self._get_unresolved_tool_uses(session)

            if not unresolved:
                # Need LLM call
                try:
                    response = await self._llm.complete(
                        messages=session.get_messages_for_llm(),
                        model=frame.model,
                        system=frame.system_prompt,
                        tools=tool_defs or None,
                        max_tokens=4096,
                    )
                except Exception as e:
                    logger.error("Interactive turn LLM error: %s", e)
                    return TurnResult(TurnAction.ERROR, text=f"LLM error: {e}")

                session.add_assistant_message(response.message.content)

                # Log assistant message
                if session_manager and agent_session_id:
                    await self._log_assistant_message(
                        session_manager,
                        agent_session_id,
                        response.message.content,
                        frame.iteration,
                    )

                tool_uses = response.message.get_tool_uses()
                if not tool_uses:
                    # Text response — send to user, pause
                    text = response.message.get_text() or ""
                    return TurnResult(TurnAction.SEND_TEXT, text=text)

                unresolved = tool_uses

            # Execute tools
            for tool_use in unresolved:
                if tool_use.name == "complete":
                    result_text = tool_use.input.get("result", "")
                    # Add tool result so session is well-formed
                    session.add_tool_result(tool_use.id, result_text, is_error=False)
                    return TurnResult(TurnAction.COMPLETE, text=result_text)

                if tool_use.name == "interrupt":
                    prompt = tool_use.input.get("prompt", "Checkpoint reached")
                    return TurnResult(TurnAction.INTERRUPT, text=prompt)

                # Check tool whitelist
                if frame.effective_tools and tool_use.name not in frame.effective_tools:
                    session.add_tool_result(
                        tool_use.id,
                        f"Tool '{tool_use.name}' is not available to this agent",
                        is_error=True,
                    )
                    continue

                try:
                    per_tool_context = ToolContext(
                        session_id=tool_context.session_id,
                        user_id=tool_context.user_id,
                        chat_id=tool_context.chat_id,
                        thread_id=tool_context.thread_id,
                        provider=tool_context.provider,
                        metadata=dict(tool_context.metadata),
                        env=dict(tool_context.env),
                        session_manager=session_manager,
                        tool_use_id=tool_use.id,
                    )
                    result = await self._tools.execute(
                        tool_use.name, tool_use.input, per_tool_context
                    )
                    session.add_tool_result(
                        tool_use.id, result.content, is_error=result.is_error
                    )
                    # Log tool result
                    if session_manager and agent_session_id:
                        await session_manager.add_tool_result(
                            tool_use_id=tool_use.id,
                            output=result.content,
                            success=not result.is_error,
                            agent_session_id=agent_session_id,
                        )
                except ChildActivated as ca:
                    # Parent paused — tool_use has no result yet
                    return TurnResult(
                        TurnAction.CHILD_ACTIVATED, child_frame=ca.child_frame
                    )
                except Exception as e:
                    logger.error("Interactive turn tool error: %s", e)
                    session.add_tool_result(
                        tool_use.id, f"Tool error: {e}", is_error=True
                    )
                    if session_manager and agent_session_id:
                        await session_manager.add_tool_result(
                            tool_use_id=tool_use.id,
                            output=f"Tool error: {e}",
                            success=False,
                            agent_session_id=agent_session_id,
                        )

        return TurnResult(TurnAction.MAX_ITERATIONS)
