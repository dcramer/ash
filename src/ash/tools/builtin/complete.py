"""Complete tool for signaling subagent task completion."""

from typing import Any

from ash.tools.base import Tool, ToolContext, ToolResult


class CompleteTool(Tool):
    """Signal that the subagent's task is complete and return a result.

    This tool is intercepted by execute_turn before reaching execute().
    It should only be available to interactive subagents, not the main agent
    or plan agent.
    """

    @property
    def name(self) -> str:
        return "complete"

    @property
    def description(self) -> str:
        return (
            "Signal that your task is complete and return a result to the caller. "
            "Your result text will be passed back as the final output."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": "Final result or summary of what was accomplished",
                },
            },
            "required": ["result"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        # This tool should be intercepted by execute_turn before reaching here.
        # If we get here, something is wrong.
        return ToolResult.error(
            "Complete tool was not intercepted by executor. "
            "This indicates the tool is being used outside an interactive subagent."
        )
