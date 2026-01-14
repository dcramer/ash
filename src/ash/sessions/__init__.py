"""JSONL-based session management for conversation persistence.

This module provides persistent session storage using JSONL files:
- context.jsonl: Full LLM context (messages, tool uses, tool results, compaction)
- history.jsonl: Human-readable conversation log (messages only)

Sessions are stored in ~/.ash/sessions/{session_key}/ directories.
"""

from ash.sessions.manager import SessionManager
from ash.sessions.reader import SessionReader, format_timestamp
from ash.sessions.types import (
    CompactionEntry,
    Entry,
    MessageEntry,
    SessionHeader,
    SessionState,
    ToolResultEntry,
    ToolUseEntry,
    session_key,
)
from ash.sessions.utils import (
    DEFAULT_RECENCY_WINDOW,
    content_block_from_dict,
    content_block_to_dict,
    fit_messages_to_budget,
    prune_messages_to_budget,
    validate_tool_pairs,
)
from ash.sessions.writer import SessionWriter

__all__ = [
    "CompactionEntry",
    "DEFAULT_RECENCY_WINDOW",
    "Entry",
    "MessageEntry",
    "SessionHeader",
    "SessionManager",
    "SessionReader",
    "SessionState",
    "SessionWriter",
    "ToolResultEntry",
    "ToolUseEntry",
    "content_block_from_dict",
    "content_block_to_dict",
    "fit_messages_to_budget",
    "format_timestamp",
    "prune_messages_to_budget",
    "session_key",
    "validate_tool_pairs",
]
