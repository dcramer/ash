"""Tool execution with logging and error handling."""

import logging
import time
from collections.abc import Callable
from typing import Any

from ash.tools.base import Tool, ToolContext, ToolResult
from ash.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Type for tool execution callbacks
ExecutionCallback = Callable[[str, dict[str, Any], ToolResult, int], None]


def _summarize_input(tool_name: str, input_data: dict[str, Any]) -> str:
    """Create a concise summary of tool input for logging.

    Args:
        tool_name: Name of the tool.
        input_data: Tool input data.

    Returns:
        Short summary string suitable for log output.
    """
    # Tool-specific summaries for common tools
    if tool_name == "write_file":
        path = input_data.get("file_path", "?")
        content = input_data.get("content", "")
        lines = content.count("\n") + 1 if content else 0
        return f"{path}, {lines} lines"

    if tool_name == "read_file":
        return input_data.get("file_path", "?")

    if tool_name == "bash":
        cmd = input_data.get("command", "")
        return cmd[:50] + "..." if len(cmd) > 50 else cmd

    if tool_name == "remember":
        content = input_data.get("content", "")
        return content[:40] + "..." if len(content) > 40 else content

    if tool_name == "recall":
        query = input_data.get("query", "")
        return query[:40] + "..." if len(query) > 40 else query

    if tool_name == "use_skill":
        return input_data.get("skill_name", "?")

    if tool_name == "web_search":
        return input_data.get("query", "?")[:40]

    if tool_name == "web_fetch":
        return input_data.get("url", "?")[:50]

    # Generic fallback: list keys
    if input_data:
        keys = list(input_data.keys())[:3]
        return ", ".join(keys)

    return ""


class ToolExecutor:
    """Execute tools with logging, timing, and error handling."""

    def __init__(
        self,
        registry: ToolRegistry,
        on_execution: ExecutionCallback | None = None,
    ):
        """Initialize executor.

        Args:
            registry: Tool registry.
            on_execution: Optional callback after each execution.
        """
        self._registry = registry
        self._on_execution = on_execution

    async def execute(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        """Execute a tool by name.

        Args:
            tool_name: Name of tool to execute.
            input_data: Tool input.
            context: Execution context.

        Returns:
            Tool result.
        """
        context = context or ToolContext()

        # Get tool
        try:
            tool = self._registry.get(tool_name)
        except KeyError:
            logger.error(f"Tool not found: {tool_name}")
            return ToolResult.error(f"Tool '{tool_name}' not found")

        # Execute with timing
        start_time = time.monotonic()
        try:
            result = await tool.execute(input_data, context)
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool_name}")
            result = ToolResult.error(f"Tool execution failed: {e}")

        duration_ms = int((time.monotonic() - start_time) * 1000)

        # Log execution - single source of truth for tool logging
        input_summary = _summarize_input(tool_name, input_data)
        if result.is_error:
            logger.error(
                f"Tool: {tool_name} | {input_summary} | failed: {result.content[:200]}"
            )
        else:
            # Call + timing at INFO, result at DEBUG
            logger.info(f"Tool: {tool_name} | {input_summary} | {duration_ms}ms")
            logger.debug(f"Tool {tool_name} result: {result.content[:200]}")

        # Callback
        if self._on_execution:
            try:
                self._on_execution(tool_name, input_data, result, duration_ms)
            except Exception:
                logger.exception("Execution callback failed")

        return result

    async def execute_tool_use(
        self,
        tool_use_id: str,
        tool_name: str,
        input_data: dict[str, Any],
        context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Execute a tool and return result in LLM format.

        Args:
            tool_use_id: Tool use ID from LLM.
            tool_name: Tool name.
            input_data: Tool input.
            context: Execution context.

        Returns:
            Dict formatted for LLM tool result.
        """
        result = await self.execute(tool_name, input_data, context)

        return {
            "tool_use_id": tool_use_id,
            "content": result.content,
            "is_error": result.is_error,
        }

    def get_tool(self, name: str) -> Tool:
        """Get a tool by name.

        Args:
            name: Tool name.

        Returns:
            Tool instance.

        Raises:
            KeyError: If tool not found.
        """
        return self._registry.get(name)

    @property
    def available_tools(self) -> list[str]:
        """Get list of available tool names."""
        return self._registry.names

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions for LLM.

        Returns:
            List of tool definitions.
        """
        return self._registry.get_definitions()
