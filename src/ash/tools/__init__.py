"""Tool system for agent capabilities."""

from ash.tools.base import Tool, ToolContext, ToolResult, build_sandbox_manager_config
from ash.tools.builtin import (
    BashTool,
    ReadFileTool,
    WebFetchTool,
    WebSearchTool,
    WriteFileTool,
)

# UseAgentTool is not exported here to avoid circular imports
# Import directly from ash.tools.builtin.agents where needed
from ash.tools.executor import ToolExecutor
from ash.tools.registry import ToolRegistry
from ash.tools.summarization import ToolResultSummarizer, create_summarizer_from_config
from ash.tools.truncation import TruncationResult, truncate_head, truncate_tail

__all__ = [
    # Base
    "Tool",
    "ToolContext",
    "ToolResult",
    "build_sandbox_manager_config",
    # Registry & Executor
    "ToolExecutor",
    "ToolRegistry",
    # Truncation & Summarization
    "TruncationResult",
    "truncate_head",
    "truncate_tail",
    "ToolResultSummarizer",
    "create_summarizer_from_config",
    # Built-in tools
    "BashTool",
    "ReadFileTool",
    "WebFetchTool",
    "WebSearchTool",
    "WriteFileTool",
]
