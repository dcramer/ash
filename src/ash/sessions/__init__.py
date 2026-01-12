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
    ToolResultEntry,
    ToolUseEntry,
    session_key,
)
from ash.sessions.utils import content_block_to_dict, validate_tool_pairs
from ash.sessions.writer import SessionWriter

__all__ = [
    "CompactionEntry",
    "Entry",
    "MessageEntry",
    "SessionHeader",
    "SessionManager",
    "SessionReader",
    "SessionWriter",
    "ToolResultEntry",
    "ToolUseEntry",
    "content_block_to_dict",
    "format_timestamp",
    "session_key",
    "validate_tool_pairs",
]
