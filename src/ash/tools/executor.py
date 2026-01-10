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

        # Log execution
        log_level = logging.ERROR if result.is_error else logging.DEBUG
        logger.log(
            log_level,
            f"Tool {tool_name} executed in {duration_ms}ms (error={result.is_error})",
        )

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
