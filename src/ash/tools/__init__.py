"""Tool system for agent capabilities."""

from ash.tools.base import Tool, ToolContext, ToolResult
from ash.tools.builtin import BashTool, WebSearchTool
from ash.tools.executor import ToolExecutor
from ash.tools.registry import ToolRegistry

__all__ = [
    # Base
    "Tool",
    "ToolContext",
    "ToolResult",
    # Registry & Executor
    "ToolExecutor",
    "ToolRegistry",
    # Built-in tools
    "BashTool",
    "WebSearchTool",
]
