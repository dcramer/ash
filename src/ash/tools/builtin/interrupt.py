"""Interrupt tool for pausing agent execution and requesting user input."""

from typing import Any

from ash.tools.base import Tool, ToolContext, ToolResult


class InterruptTool(Tool):
    """Pause execution and request user input.

    This tool allows agents to checkpoint their state and pause execution
    to request user input. The agent executor detects interrupt tool calls
    and handles them specially by saving state and returning to the caller.
    """

    @property
    def name(self) -> str:
        return "interrupt"

    @property
    def description(self) -> str:
        return (
            "Pause execution and request user input. "
            "Use this to checkpoint progress and get user approval before proceeding. "
            "The user's response will be returned as the tool result when execution resumes."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Message to show the user explaining what you've done and what you're asking for",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional suggested responses (e.g., ['Proceed', 'Cancel', 'Modify'])",
                },
            },
            "required": ["prompt"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        # This tool should never actually execute normally.
        # The AgentExecutor intercepts interrupt calls and handles them specially.
        # If we get here, something is wrong.
        return ToolResult.error(
            "Interrupt tool was executed directly instead of being intercepted. "
            "This indicates an executor configuration issue."
        )
