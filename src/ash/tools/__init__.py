"""Tool system for agent capabilities."""

from ash.tools.base import Tool, ToolContext, ToolResult
from ash.tools.builtin import (
    BashTool,
    FileAccessTracker,
    ReadFileTool,
    WebFetchTool,
    WebSearchTool,
    WriteFileTool,
)
from ash.tools.builtin.memory import RecallTool, RememberTool
from ash.tools.builtin.skills import UseSkillTool, WriteSkillTool
from ash.tools.executor import ToolExecutor
from ash.tools.registry import ToolRegistry
from ash.tools.summarization import ToolResultSummarizer, create_summarizer_from_config
from ash.tools.truncation import TruncationResult, truncate_head, truncate_tail

__all__ = [
    # Base
    "Tool",
    "ToolContext",
    "ToolResult",
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
    "FileAccessTracker",
    "ReadFileTool",
    "WebFetchTool",
    "WebSearchTool",
    "WriteFileTool",
    # Memory tools
    "RecallTool",
    "RememberTool",
    # Skill tools
    "UseSkillTool",
    "WriteSkillTool",
]
