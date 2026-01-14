"""Agent invocation tool."""

import logging
from typing import TYPE_CHECKING, Any

from ash.agents.base import AgentContext
from ash.tools.base import Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from ash.agents import AgentExecutor, AgentRegistry
    from ash.config import AshConfig
    from ash.skills import SkillRegistry

logger = logging.getLogger(__name__)


class UseAgentTool(Tool):
    """Invoke a built-in agent for complex tasks.

    Agents run in isolated subagent loops with their own
    system prompts and tool restrictions. Use agents for
    complex multi-step tasks that benefit from focused execution.
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

    @property
    def name(self) -> str:
        """Tool name."""
        return "use_agent"

    @property
    def description(self) -> str:
        """Tool description."""
        agents = self._registry.list_available()
        agent_list = ", ".join(a.config.name for a in agents)
        return f"Run a specialized agent for complex tasks. Available: {agent_list}"

    @property
    def input_schema(self) -> dict[str, Any]:
        """Input schema for the tool."""
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
            },
            "required": ["agent", "message"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        """Execute the tool.

        Args:
            input_data: Tool input with agent name and message.
            context: Optional tool execution context.

        Returns:
            ToolResult with agent output.
        """
        agent_name = input_data.get("agent")
        message = input_data.get("message")
        extra_input = input_data.get("input", {})

        if not agent_name:
            return ToolResult.error("Missing required field: agent")

        if not message:
            return ToolResult.error("Missing required field: message")

        # Look up agent
        if not self._registry.has(agent_name):
            available = ", ".join(self._registry.list_names())
            return ToolResult.error(
                f"Agent '{agent_name}' not found. Available: {available}"
            )

        agent = self._registry.get(agent_name)

        # Build context
        agent_context = AgentContext(
            session_id=context.session_id if context else None,
            user_id=context.user_id if context else None,
            chat_id=context.chat_id if context else None,
            input_data=extra_input,
        )

        # Execute agent
        result = await self._executor.execute(agent, message, agent_context)

        # Reload skills after skill-writer completes successfully
        if agent_name == "skill-writer" and not result.is_error:
            if self._skill_registry and self._config:
                count = self._skill_registry.reload_workspace(self._config.workspace)
                if count > 0:
                    logger.info(f"Reloaded {count} new skill(s) after skill-writer")

        if result.is_error:
            return ToolResult.error(result.content)

        return ToolResult.success(result.content)
