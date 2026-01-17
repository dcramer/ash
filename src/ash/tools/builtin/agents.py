"""Agent invocation tool."""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from ash.agents.base import AgentContext, CheckpointState
from ash.agents.executor import is_cancel_message
from ash.tools.base import Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from ash.agents import AgentExecutor, AgentRegistry
    from ash.config import AshConfig
    from ash.skills import SkillRegistry

logger = logging.getLogger(__name__)

# Metadata key for checkpoint data in tool results
CHECKPOINT_METADATA_KEY = "checkpoint"


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
        skill_registry: "SkillRegistry | None" = None,
        config: "AshConfig | None" = None,
    ) -> None:
        """Initialize the tool.

        Args:
            registry: Agent registry to look up agents.
            executor: Agent executor to run agents.
            skill_registry: Optional skill registry for reloading after skill-writer.
            config: Optional config for workspace path.
        """
        self._registry = registry
        self._executor = executor
        self._skill_registry = skill_registry
        self._config = config
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
                context, input_data=extra_input
            )
        else:
            agent_context = AgentContext(input_data=extra_input)

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

        result = await self._executor.execute(
            agent,
            message,
            agent_context,
            resume_from=resume_from,
            user_response=checkpoint_response,
        )

        # Handle interrupted result (checkpoint)
        if result.checkpoint:
            checkpoint = result.checkpoint
            await self.store_checkpoint(checkpoint)

            # Build response with checkpoint info
            options_str = ""
            if checkpoint.options:
                options_str = (
                    f"\n\nSuggested responses: {', '.join(checkpoint.options)}"
                )

            return ToolResult.success(
                f"**Agent paused for input**\n\n{checkpoint.prompt}{options_str}",
                **{CHECKPOINT_METADATA_KEY: checkpoint.to_dict()},
            )

        # Handle skill agent completion (reload skills)
        if agent.config.is_skill_agent and not result.is_error:
            if self._skill_registry and self._config:
                count = self._skill_registry.reload_workspace(self._config.workspace)
                if count > 0:
                    logger.info(f"Reloaded {count} new skill(s) after {agent_name}")

        if result.is_error:
            return ToolResult.error(result.content)

        return ToolResult.success(result.content)
