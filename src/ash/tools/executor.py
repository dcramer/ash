"""Tool execution with logging and error handling."""

import logging
import time
from collections.abc import Callable
from typing import Any

from ash.llm.types import ToolDefinition
from ash.logging import log_context
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
        metadata = context.metadata if isinstance(context.metadata, dict) else {}
        chat_type = metadata.get("chat_type")
        source_username = metadata.get("username") or metadata.get("source_username")

        with log_context(
            chat_id=context.chat_id,
            session_id=context.session_id,
            provider=context.provider,
            user_id=context.user_id,
            thread_id=context.thread_id,
            chat_type=str(chat_type) if chat_type else None,
            source_username=(
                str(source_username) if source_username is not None else None
            ),
        ):
            # Check per-session tool overrides before global registry
            tool = context.tool_overrides.get(tool_name)
            if tool is None:
                try:
                    tool = self._registry.get(tool_name)
                except KeyError:
                    logger.error(
                        "tool_not_found",
                        extra={"gen_ai.tool.name": tool_name, "error.type": "KeyError"},
                    )
                    return ToolResult.error(f"Tool '{tool_name}' not found")

            logger.debug(f"Tool {tool_name} input: {input_data}")

            start_time = time.monotonic()
            try:
                result = await tool.execute(input_data, context)
            except Exception as e:
                logger.exception(
                    "tool_execution_failed", extra={"gen_ai.tool.name": tool_name}
                )
                result = ToolResult.error(f"Tool execution failed: {e}")

            duration_ms = int((time.monotonic() - start_time) * 1000)

            exit_code = result.metadata.get("exit_code") if result.metadata else None

            log_extra: dict[str, Any] = {
                "gen_ai.tool.name": tool_name,
                "gen_ai.tool.call.arguments": input_data,
                "duration_ms": duration_ms,
            }
            if exit_code is not None:
                log_extra["process.exit_code"] = exit_code

            if result.is_error:
                log_extra["error.message"] = result.content[:500]
                logger.error(
                    "tool_executed",
                    extra=log_extra,
                )
            elif exit_code is not None and exit_code != 0:
                log_extra["output.preview"] = _truncate(result.content, 200)
                logger.warning(
                    "tool_executed",
                    extra=log_extra,
                )
            else:
                logger.info(
                    "tool_executed",
                    extra=log_extra,
                )
                logger.debug(f"Tool {tool_name} result: {result.content[:200]}")

            if self._on_execution:
                try:
                    self._on_execution(tool_name, input_data, result, duration_ms)
                except Exception:
                    logger.warning("execution_callback_failed", exc_info=True)

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
