"""Tool execution with logging and error handling."""

import logging
import time
from collections.abc import Callable
from typing import Any

from ash.llm.types import ToolDefinition
from ash.tools.base import Tool, ToolContext, ToolResult
from ash.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Type for tool execution callbacks
ExecutionCallback = Callable[[str, dict[str, Any], ToolResult, int], None]


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# Tool input summarizers: each returns a summary string from input_data
_TOOL_SUMMARIZERS: dict[str, Any] = {
    "write_file": lambda d: f"{d.get('file_path', '?')}, {d.get('content', '').count(chr(10)) + 1 if d.get('content') else 0} lines",
    "read_file": lambda d: d.get("file_path", "?"),
    "bash": lambda d: _truncate(d.get("command", ""), 50),
    "use_agent": lambda d: d.get("agent", "?"),
    "use_skill": lambda d: f"{d.get('skill', '?')}: {_truncate(d.get('message', ''), 40)}",
    "web_search": lambda d: _truncate(d.get("query", "?"), 40),
    "web_fetch": lambda d: _truncate(d.get("url", "?"), 50),
}


def _summarize_input(tool_name: str, input_data: dict[str, Any]) -> str:
    if summarizer := _TOOL_SUMMARIZERS.get(tool_name):
        return summarizer(input_data)
    if input_data:
        return ", ".join(list(input_data.keys())[:3])
    return ""


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        on_execution: ExecutionCallback | None = None,
    ):
        self._registry = registry
        self._on_execution = on_execution

    async def execute(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        context = context or ToolContext()

        # Check per-session tool overrides before global registry
        tool = context.tool_overrides.get(tool_name)
        if tool is None:
            try:
                tool = self._registry.get(tool_name)
            except KeyError:
                logger.error(f"Tool not found: {tool_name}")
                return ToolResult.error(f"Tool '{tool_name}' not found")

        logger.debug(f"Tool {tool_name} input: {input_data}")

        start_time = time.monotonic()
        try:
            result = await tool.execute(input_data, context)
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool_name}")
            result = ToolResult.error(f"Tool execution failed: {e}")

        duration_ms = int((time.monotonic() - start_time) * 1000)

        input_summary = _summarize_input(tool_name, input_data)
        exit_code = result.metadata.get("exit_code") if result.metadata else None

        if result.is_error:
            logger.error(
                f"Tool: {tool_name} | {input_summary} | failed: {result.content[:200]}"
            )
        elif exit_code is not None and exit_code != 0:
            output_preview = _truncate(result.content, 100)
            logger.warning(
                f"Tool: {tool_name} | {input_summary} | exit={exit_code} | {output_preview}"
            )
        else:
            logger.info(f"Tool: {tool_name} | {input_summary} | {duration_ms}ms")
            logger.debug(f"Tool {tool_name} result: {result.content[:200]}")

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
        result = await self.execute(tool_name, input_data, context)
        return {
            "tool_use_id": tool_use_id,
            "content": result.content,
            "is_error": result.is_error,
        }

    def get_tool(self, name: str) -> Tool:
        return self._registry.get(name)

    @property
    def available_tools(self) -> list[str]:
        return self._registry.names

    def get_definitions(self) -> list[ToolDefinition]:
        return self._registry.get_definitions()
