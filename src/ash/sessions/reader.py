"""JSONL session reader for loading context."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles

from ash.llm.types import (
    ContentBlock,
    Message,
    Role,
    TextContent,
    ToolResult,
    ToolUse,
)
from ash.sessions.types import (
    CompactionEntry,
    Entry,
    MessageEntry,
    SessionHeader,
    ToolResultEntry,
    ToolUseEntry,
    parse_entry,
)
from ash.sessions.utils import DEFAULT_RECENCY_WINDOW, prune_messages_to_budget

logger = logging.getLogger(__name__)


def format_timestamp(dt: datetime) -> str:
    """Format datetime for message prefix.

    Args:
        dt: Datetime to format.

    Returns:
        Formatted string like '[2026-01-11 10:30:45]'.
    """
    return dt.strftime("[%Y-%m-%d %H:%M:%S]")


class SessionReader:
    """Reads session entries from JSONL files."""

    def __init__(self, session_dir: Path) -> None:
        """Initialize session reader.

        Args:
            session_dir: Directory containing session files.
        """
        self.session_dir = session_dir
        self.context_file = session_dir / "context.jsonl"
        self.history_file = session_dir / "history.jsonl"

    def exists(self) -> bool:
        """Check if session exists.

        Returns:
            True if context.jsonl exists.
        """
        return self.context_file.exists()

    async def load_entries(self) -> list[Entry]:
        """Load all entries from context.jsonl.

        Returns:
            List of parsed entry objects.
        """
        if not self.context_file.exists():
            return []

        entries: list[Entry] = []
        async with aiofiles.open(self.context_file, encoding="utf-8") as f:
            line_num = 0
            async for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(parse_entry(data))
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        "Failed to parse line %d in %s: %s",
                        line_num,
                        self.context_file,
                        e,
                    )
        return entries

    async def load_header(self) -> SessionHeader | None:
        """Load just the session header.

        Returns:
            Session header or None if not found.
        """
        entries = await self.load_entries()
        for entry in entries:
            if isinstance(entry, SessionHeader):
                return entry
        return None

    async def load_messages_for_llm(
        self,
        token_budget: int | None = None,
        recency_window: int = DEFAULT_RECENCY_WINDOW,
        include_timestamps: bool = False,
    ) -> tuple[list[Message], list[str]]:
        """Load messages formatted for LLM API, with token-aware pruning.

        This reconstructs the conversation from entries, combining messages
        with their tool uses and results into proper API format.

        Args:
            token_budget: Maximum tokens for messages (None = no limit).
            recency_window: Always keep at least this many recent messages.
            include_timestamps: Whether to prefix messages with timestamps.

        Returns:
            Tuple of (messages for LLM, message IDs for deduplication).
        """
        entries = await self.load_entries()

        # Build messages from entries
        messages, message_ids, token_counts = self._build_messages(
            entries, include_timestamps=include_timestamps
        )

        # Use shared pruning logic
        return prune_messages_to_budget(
            messages,
            token_counts,
            token_budget,
            recency_window,
            message_ids,
        )

    def _build_messages(
        self,
        entries: list[Entry],
        include_timestamps: bool = False,
    ) -> tuple[list[Message], list[str], list[int]]:
        """Build Message objects from entries.

        Groups tool uses and results with their parent messages.

        Args:
            entries: List of parsed entries.
            include_timestamps: Whether to prefix messages with timestamps.

        Returns:
            Tuple of (messages, message_ids, token_counts).
        """
        from ash.core.tokens import estimate_message_tokens

        messages: list[Message] = []
        message_ids: list[str] = []
        token_counts: list[int] = []
        pending_results: list[ToolResult] = []

        def flush_pending_results() -> None:
            """Flush pending tool results as a user message."""
            if not pending_results:
                return
            messages.append(Message(role=Role.USER, content=list(pending_results)))
            message_ids.append("")
            token_counts.append(
                estimate_message_tokens(
                    "user",
                    [
                        {"type": "tool_result", "content": r.content}
                        for r in pending_results
                    ],
                )
            )
            pending_results.clear()

        for entry in entries:
            match entry:
                case SessionHeader() | ToolUseEntry() | CompactionEntry():
                    # Skip: header, tool uses (embedded in messages), compaction markers
                    pass

                case MessageEntry():
                    flush_pending_results()

                    content = self._convert_content(entry.content)
                    if include_timestamps and entry.created_at:
                        content = self._prefix_with_timestamp(content, entry.created_at)

                    messages.append(Message(role=Role(entry.role), content=content))
                    message_ids.append(entry.id)
                    token_counts.append(
                        entry.token_count
                        if entry.token_count is not None
                        else estimate_message_tokens(entry.role, entry.content)
                    )

                case ToolResultEntry():
                    pending_results.append(
                        ToolResult(
                            tool_use_id=entry.tool_use_id,
                            content=entry.output,
                            is_error=not entry.success,
                        )
                    )

        flush_pending_results()
        return messages, message_ids, token_counts

    def _convert_content(
        self, content: str | list[dict[str, Any]]
    ) -> str | list[ContentBlock]:
        """Convert stored content to Message content format.

        Args:
            content: Stored content (string or list of block dicts).

        Returns:
            Content for Message.
        """
        if isinstance(content, str):
            return content

        blocks: list[ContentBlock] = []
        for block in content:
            match block.get("type"):
                case "text":
                    blocks.append(TextContent(text=block["text"]))
                case "tool_use":
                    blocks.append(
                        ToolUse(
                            id=block["id"],
                            name=block["name"],
                            input=block["input"],
                        )
                    )
                case "tool_result":
                    blocks.append(
                        ToolResult(
                            tool_use_id=block["tool_use_id"],
                            content=block["content"],
                            is_error=block.get("is_error", False),
                        )
                    )
        return blocks if blocks else ""

    def _prefix_with_timestamp(
        self,
        content: str | list[ContentBlock],
        timestamp: datetime,
    ) -> str | list[ContentBlock]:
        """Prefix message content with timestamp.

        For string content, prepends the timestamp.
        For block content, prepends to the first text block.

        Args:
            content: Message content.
            timestamp: Timestamp to prefix.

        Returns:
            Content with timestamp prefix.
        """
        ts_prefix = format_timestamp(timestamp) + " "

        if isinstance(content, str):
            return ts_prefix + content

        if not content:
            return ts_prefix.strip()

        # Find first text block and prepend timestamp
        result: list[ContentBlock] = []
        prefixed = False

        for block in content:
            if isinstance(block, TextContent) and not prefixed:
                result.append(TextContent(text=ts_prefix + block.text))
                prefixed = True
            else:
                result.append(block)

        # If no text block found, add one at the start
        if not prefixed:
            result.insert(0, TextContent(text=ts_prefix.strip()))

        return result

    async def get_message_ids(self) -> set[str]:
        """Get all message IDs in the session.

        Returns:
            Set of message IDs.
        """
        entries = await self.load_entries()
        return {entry.id for entry in entries if isinstance(entry, MessageEntry)}

    async def get_last_compaction(self) -> CompactionEntry | None:
        """Get the most recent compaction entry.

        Returns:
            Last compaction entry or None.
        """
        entries = await self.load_entries()
        for entry in reversed(entries):
            if isinstance(entry, CompactionEntry):
                return entry
        return None

    async def has_message_with_external_id(self, external_id: str) -> bool:
        """Check if a message with given external ID exists.

        Used to avoid processing duplicate messages (e.g., from Telegram).

        Args:
            external_id: External message ID (e.g., Telegram message ID).

        Returns:
            True if message exists, False otherwise.
        """
        entries = await self.load_entries()
        for entry in entries:
            if isinstance(entry, MessageEntry):
                if entry.metadata and entry.metadata.get("external_id") == external_id:
                    return True
        return False

    async def get_message_by_external_id(self, external_id: str) -> MessageEntry | None:
        """Find message by external ID.

        Searches both user messages (external_id) and assistant messages
        (bot_response_id) to support reply-to functionality.

        Args:
            external_id: External message ID.

        Returns:
            MessageEntry if found, None otherwise.
        """
        entries = await self.load_entries()
        for entry in entries:
            if isinstance(entry, MessageEntry) and entry.metadata:
                # Check external_id (user messages)
                if entry.metadata.get("external_id") == external_id:
                    return entry
                # Check bot_response_id (assistant messages)
                if entry.metadata.get("bot_response_id") == external_id:
                    return entry
        return None

    async def get_messages_around(
        self, message_id: str, window: int = 3
    ) -> list[MessageEntry]:
        """Get messages around a specific message.

        Returns the target message plus N messages before and after it,
        sorted chronologically.

        Args:
            message_id: Target message ID.
            window: Number of messages before and after (default 3).

        Returns:
            List of MessageEntry sorted by created_at.
        """
        entries = await self.load_entries()
        messages = [e for e in entries if isinstance(e, MessageEntry)]

        # Find the target message index
        target_idx = None
        for i, msg in enumerate(messages):
            if msg.id == message_id:
                target_idx = i
                break

        if target_idx is None:
            return []

        # Get window around target
        start = max(0, target_idx - window)
        end = min(len(messages), target_idx + window + 1)

        return messages[start:end]

    async def search_messages(self, query: str, limit: int = 20) -> list[MessageEntry]:
        """Search messages by content.

        Args:
            query: Search query (case-insensitive substring match).
            limit: Maximum number of results.

        Returns:
            List of matching MessageEntry, most recent first.
        """
        entries = await self.load_entries()
        messages = [e for e in entries if isinstance(e, MessageEntry)]
        query_lower = query.lower()

        results: list[MessageEntry] = []
        for msg in reversed(messages):  # Most recent first
            if query_lower in msg._extract_text_content().lower():
                results.append(msg)
                if len(results) >= limit:
                    break

        return results
