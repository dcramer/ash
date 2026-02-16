"""Agent invocation tool."""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from ash.agents.executor import is_cancel_message
from ash.agents.types import AgentContext, CheckpointState
from ash.tools.base import Tool, ToolContext, ToolResult, format_subagent_result

if TYPE_CHECKING:
    from ash.agents import AgentExecutor, AgentRegistry
    from ash.config import AshConfig

logger = logging.getLogger(__name__)

# Metadata key for checkpoint data in tool results
CHECKPOINT_METADATA_KEY = "checkpoint"


def format_agent_result(content: str, agent_name: str) -> str:
    """Format agent result with structured tags for LLM clarity."""

    return format_subagent_result(content, "agent", agent_name)


class UseAgentTool(Tool):
    """Invoke a built-in agent for complex tasks.

    Agents run in isolated subagent loops with their own
    system prompts and tool restrictions. Use agents for
    complex multi-step tasks that benefit from focused execution.

    Supports checkpoint/resume flow for long-running agents:
    - When an agent calls the `interrupt` tool, this returns with checkpoint metadata
    - Resume by calling with `resume_checkpoint_id` and `checkpoint_response`
    """

    def __init__(
        self,
        registry: "AgentRegistry",
        executor: "AgentExecutor",
        config: "AshConfig | None" = None,
        voice: str | None = None,
        subagent_context: str | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            registry: Agent registry to look up agents.
            executor: Agent executor to run agents.
            config: Application configuration for model resolution.
            voice: Optional communication style for user-facing subagent messages.
            subagent_context: Shared prompt context (sandbox, runtime, tool guidance) for subagents.
        """
        self._registry = registry
        self._executor = executor
        self._config = config
        self._voice = voice
        self._subagent_context = subagent_context
        # In-memory checkpoint storage (keyed by checkpoint_id)
        # In production, this would be stored in the session via SessionManager
        self._pending_checkpoints: dict[str, CheckpointState] = {}
        self._checkpoint_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "use_agent"

    @property
    def description(self) -> str:
        agents = self._registry.list_agents()
        agent_list = ", ".join(a.config.name for a in agents)
        return f"Run a specialized agent for complex tasks. Available: {agent_list}"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Name of the agent to run",
                },
                "message": {
                    "type": "string",
                    "description": "Message/task for the agent",
                },
                "input": {
                    "type": "object",
                    "description": "Additional input data for the agent (optional)",
                },
                "resume_checkpoint_id": {
                    "type": "string",
                    "description": (
                        "ID of a checkpoint to resume from. "
                        "If provided, continues a previously interrupted agent."
                    ),
                },
                "checkpoint_response": {
                    "type": "string",
                    "description": (
                        "User's response when resuming from a checkpoint. "
                        "Required when resume_checkpoint_id is provided."
                    ),
                },
            },
            "required": ["agent", "message"],
        }

    async def store_checkpoint(self, checkpoint: CheckpointState) -> None:
        """Store a checkpoint for later retrieval."""
        async with self._checkpoint_lock:
            self._pending_checkpoints[checkpoint.checkpoint_id] = checkpoint

    async def get_checkpoint(self, checkpoint_id: str) -> CheckpointState | None:
        """Retrieve a stored checkpoint, returning None if not found or expired."""
        async with self._checkpoint_lock:
            checkpoint = self._pending_checkpoints.get(checkpoint_id)
            if checkpoint is None or not checkpoint.is_expired():
                return checkpoint
            # Clean up expired checkpoint
            del self._pending_checkpoints[checkpoint_id]
            return None

    async def clear_checkpoint(self, checkpoint_id: str) -> None:
        """Remove a stored checkpoint."""
        async with self._checkpoint_lock:
            self._pending_checkpoints.pop(checkpoint_id, None)

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        agent_name = input_data.get("agent")
        message = input_data.get("message")
        extra_input = input_data.get("input", {})
        resume_checkpoint_id = input_data.get("resume_checkpoint_id")
        checkpoint_response = input_data.get("checkpoint_response")

        if not agent_name:
            return ToolResult.error("Missing required field: agent")

        if not message:
            return ToolResult.error("Missing required field: message")

        if agent_name not in self._registry:
            available = ", ".join(a.config.name for a in self._registry.list_agents())
            return ToolResult.error(
                f"Agent '{agent_name}' not found. Available: {available}"
            )

        agent = self._registry.get(agent_name)

        if context:
            agent_context = AgentContext.from_tool_context(
                context,
                input_data=extra_input,
                voice=self._voice,
                shared_prompt=self._subagent_context,
            )
        else:
            agent_context = AgentContext(
                input_data=extra_input,
                voice=self._voice,
                shared_prompt=self._subagent_context,
            )

        # Handle resume from checkpoint
        resume_from: CheckpointState | None = None
        if resume_checkpoint_id:
            if not checkpoint_response:
                return ToolResult.error(
                    "checkpoint_response is required when resume_checkpoint_id is provided"
                )

            # Check for cancel intent
            if is_cancel_message(checkpoint_response):
                await self.clear_checkpoint(resume_checkpoint_id)
                return ToolResult.success(
                    f"Agent '{agent_name}' execution cancelled by user."
                )

            resume_from = await self.get_checkpoint(resume_checkpoint_id)
            if resume_from is None:
                return ToolResult.error(
                    f"Checkpoint '{resume_checkpoint_id}' not found or expired"
                )

            # Clear the checkpoint since we're resuming (executor validates ownership)
            await self.clear_checkpoint(resume_checkpoint_id)

        # Get session info from context for subagent logging
        session_manager, tool_use_id = (
            context.get_session_info() if context else (None, None)
        )

        agent_config = agent.config

        # Checkpoint agents and passthrough agents use batch execution.
        # Non-checkpoint agents use the interactive stack.
        use_batch = (
            agent_config.supports_checkpointing
            or agent_config.is_passthrough
            or resume_from is not None
        )

        if use_batch:
            return await self._execute_batch(
                agent,
                agent_name,
                message,
                agent_context,
                resume_from=resume_from,
                checkpoint_response=checkpoint_response,
                session_manager=session_manager,
                tool_use_id=tool_use_id,
            )

        # Interactive path: build StackFrame and raise ChildActivated
        return await self._execute_interactive(
            agent,
            agent_config,
            message,
            agent_context,
            session_manager=session_manager,
            tool_use_id=tool_use_id,
        )

    async def _execute_batch(
        self,
        agent: Any,
        agent_name: str,
        message: str,
        agent_context: AgentContext,
        *,
        resume_from: CheckpointState | None = None,
        checkpoint_response: str | None = None,
        session_manager: Any = None,
        tool_use_id: str | None = None,
    ) -> ToolResult:
        """Execute agent in batch mode (checkpoint/passthrough agents)."""
        result = await self._executor.execute(
            agent,
            message,
            agent_context,
            resume_from=resume_from,
            user_response=checkpoint_response,
            session_manager=session_manager,
            parent_tool_use_id=tool_use_id,
        )

        # Handle interrupted result (checkpoint)
        if result.checkpoint:
            checkpoint = result.checkpoint
            await self.store_checkpoint(checkpoint)

            options_str = ""
            if checkpoint.options:
                options_str = (
                    f"\n\nSuggested responses: {', '.join(checkpoint.options)}"
                )

            return ToolResult.success(
                f"{checkpoint.prompt}{options_str}",
                **{CHECKPOINT_METADATA_KEY: checkpoint.to_dict()},
            )

        if result.is_error:
            return ToolResult.error(result.content)

        return ToolResult.success(format_agent_result(result.content, agent_name))

    async def _execute_interactive(
        self,
        agent: Any,
        agent_config: Any,
        message: str,
        agent_context: AgentContext,
        *,
        session_manager: Any = None,
        tool_use_id: str | None = None,
    ) -> ToolResult:
        """Build a StackFrame and raise ChildActivated for interactive execution."""
        from ash.agents.types import ChildActivated, StackFrame
        from ash.core.session import SessionState
        from ash.sessions.types import generate_id

        # Resolve model
        overrides = self._config.agents.get(agent_config.name) if self._config else None
        model_alias = (overrides.model if overrides else None) or agent_config.model
        resolved_model: str | None = None
        if model_alias and self._config:
            try:
                resolved_model = self._config.get_model(model_alias).model
            except Exception:
                logger.warning(
                    "Failed to resolve model '%s' for agent '%s'",
                    model_alias,
                    agent_config.name,
                )

        # Start agent session for logging
        agent_session_id: str | None = None
        if session_manager and tool_use_id:
            agent_session_id = await session_manager.start_agent_session(
                parent_tool_use_id=tool_use_id,
                agent_type="agent",
                agent_name=agent_config.name,
            )

        # Build child session with initial message
        child_session = SessionState(
            session_id=f"agent-{agent_config.name}-{agent_context.session_id or 'unknown'}",
            provider=agent_context.provider or "",
            chat_id=agent_context.chat_id or "",
            user_id=agent_context.user_id or "",
        )
        child_session.add_user_message(message)

        system_prompt = agent.build_system_prompt(agent_context)

        child_frame = StackFrame(
            frame_id=generate_id(),
            agent_name=agent_config.name,
            agent_type="agent",
            session=child_session,
            system_prompt=system_prompt,
            context=agent_context,
            model=resolved_model,
            max_iterations=agent_config.max_iterations,
            effective_tools=agent_config.get_effective_tools(),
            voice=self._voice,
            parent_tool_use_id=tool_use_id,
            agent_session_id=agent_session_id,
        )

        raise ChildActivated(child_frame)
