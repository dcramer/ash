"""Telegram message handlers.

This package provides message handling for Telegram provider.

Public API:
- TelegramMessageHandler: Main message handler class
- CheckpointHandler: Handles checkpoint storage and callback resumption
- ToolTracker: Tracks tool calls and manages thinking messages
- ProgressMessageTool: Per-run send_message tool for progress updates

Utilities:
- escape_markdown_v2: Escape text for MarkdownV2 format
- format_thinking_status: Format thinking status line
- format_tool_summary: Format tool call summary
- format_tool_brief: Format individual tool execution
"""

from ash.providers.telegram.handlers.checkpoint_handler import CheckpointHandler
from ash.providers.telegram.handlers.message_handler import TelegramMessageHandler
from ash.providers.telegram.handlers.session_handler import (
    SessionContext,
    SessionHandler,
)
from ash.providers.telegram.handlers.tool_tracker import (
    ProgressMessageTool,
    ToolTracker,
)
from ash.providers.telegram.handlers.utils import (
    MAX_MESSAGE_LENGTH,
    MIN_EDIT_INTERVAL,
    STREAM_DELAY,
    escape_markdown_v2,
    extract_text_content,
    format_thinking_status,
    format_tool_brief,
    format_tool_summary,
    get_domain,
    get_filename,
    resolve_agent_model,
    resolve_skill_model,
    truncate_str,
)

__all__ = [
    # Main classes
    "CheckpointHandler",
    "ProgressMessageTool",
    "SessionContext",
    "SessionHandler",
    "TelegramMessageHandler",
    "ToolTracker",
    # Constants
    "MAX_MESSAGE_LENGTH",
    "MIN_EDIT_INTERVAL",
    "STREAM_DELAY",
    # Formatting functions
    "escape_markdown_v2",
    "format_thinking_status",
    "format_tool_brief",
    "format_tool_summary",
    # Utility functions
    "extract_text_content",
    "get_domain",
    "get_filename",
    "resolve_agent_model",
    "resolve_skill_model",
    "truncate_str",
]
