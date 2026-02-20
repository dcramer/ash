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
    return dt.strftime("[%Y-%m-%d %H:%M:%S]")


class SessionReader:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.context_file = session_dir / "context.jsonl"
        self.history_file = session_dir / "history.jsonl"

    def exists(self) -> bool:
        return self.context_file.exists()

    async def load_entries(self) -> list[Entry]:
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
                    entries.append(parse_entry(json.loads(line)))
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        "context_parse_error",
                        extra={
                            "line_num": line_num,
                            "file": str(self.context_file),
                            "error.message": str(e),
                        },
                    )
        return entries

    async def load_header(self) -> SessionHeader | None:
        for entry in await self.load_entries():
            if isinstance(entry, SessionHeader):
                return entry
        return None

    async def load_messages_for_llm(
        self,
        token_budget: int | None = None,
        recency_window: int = DEFAULT_RECENCY_WINDOW,
        include_timestamps: bool = False,
    ) -> tuple[list[Message], list[str]]:
        entries = await self.load_entries()
        messages, message_ids, token_counts = self._build_messages(
            entries, include_timestamps=include_timestamps
        )
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
        from ash.core.tokens import estimate_message_tokens

        tool_use_ids = self._collect_tool_use_ids(entries)
        messages: list[Message] = []
        message_ids: list[str] = []
        token_counts: list[int] = []
        pending_results: list[ToolResult] = []

        def flush_pending_results() -> None:
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
            if isinstance(entry, (SessionHeader, ToolUseEntry, CompactionEntry)):
                continue

            if isinstance(entry, MessageEntry):
                # Backward compat: skip subagent messages from legacy sessions
                # that wrote them into the parent's context.jsonl
                if entry.agent_session_id is not None:
                    continue
                flush_pending_results()
                content = self._convert_content(entry.content)
                if not content:
                    continue
                if include_timestamps and entry.created_at:
                    content = self._prefix_with_timestamp(content, entry.created_at)

                messages.append(Message(role=Role(entry.role), content=content))
                message_ids.append(entry.id)
                token_counts.append(
                    entry.token_count
                    if entry.token_count is not None
                    else estimate_message_tokens(entry.role, entry.content)
                )

            elif (
                isinstance(entry, ToolResultEntry)
                and entry.tool_use_id in tool_use_ids
                and entry.agent_session_id is None  # Skip legacy subagent results
            ):
                pending_results.append(
                    ToolResult(
                        tool_use_id=entry.tool_use_id,
                        content=entry.output,
                        is_error=not entry.success,
                    )
                )

        flush_pending_results()
        return messages, message_ids, token_counts

    def _collect_tool_use_ids(self, entries: list[Entry]) -> set[str]:
        tool_use_ids: set[str] = set()
        for entry in entries:
            if isinstance(entry, ToolUseEntry) and entry.agent_session_id is None:
                tool_use_ids.add(entry.id)
            elif (
                isinstance(entry, MessageEntry)
                and entry.agent_session_id is None
                and isinstance(entry.content, list)
            ):
                for block in entry.content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_use_ids.add(block["id"])
        return tool_use_ids

    def _convert_content(
        self, content: str | list[dict[str, Any]]
    ) -> str | list[ContentBlock]:
        if isinstance(content, str):
            return content

        from ash.sessions.utils import content_block_from_dict

        blocks: list[ContentBlock] = [
            block
            for item in content
            if (block := content_block_from_dict(item)) is not None
        ]
        return blocks if blocks else ""

    def _prefix_with_timestamp(
        self,
        content: str | list[ContentBlock],
        timestamp: datetime,
    ) -> str | list[ContentBlock]:
        ts_prefix = format_timestamp(timestamp) + " "

        if isinstance(content, str):
            return ts_prefix + content

        if not content:
            return ts_prefix.strip()

        result: list[ContentBlock] = []
        prefixed = False
        for block in content:
            if isinstance(block, TextContent) and not prefixed:
                result.append(TextContent(text=ts_prefix + block.text))
                prefixed = True
            else:
                result.append(block)

        if not prefixed:
            result.insert(0, TextContent(text=ts_prefix.strip()))

        return result

    async def get_message_ids(self) -> set[str]:
        entries = await self.load_entries()
        return {entry.id for entry in entries if isinstance(entry, MessageEntry)}

    async def get_last_compaction(self) -> CompactionEntry | None:
        for entry in reversed(await self.load_entries()):
            if isinstance(entry, CompactionEntry):
                return entry
        return None

    async def has_message_with_external_id(self, external_id: str) -> bool:
        """Check if a message with the given external ID exists in this session.

        This checks BOTH user messages (stored with external_id) AND bot responses
        (stored with bot_response_id). This is critical for reply detection in group
        chats - when a user replies to a bot message, we need to find the bot's
        response by its Telegram message ID (stored as bot_response_id).

        Without checking bot_response_id, replies to bot messages would be skipped
        because _should_skip_reply() wouldn't find the reply target.
        """
        for entry in await self.load_entries():
            if isinstance(entry, MessageEntry) and entry.metadata:
                if external_id in (
                    entry.metadata.get("external_id"),
                    entry.metadata.get("bot_response_id"),
                ):
                    return True
        return False

    async def get_message_by_external_id(self, external_id: str) -> MessageEntry | None:
        for entry in await self.load_entries():
            if isinstance(entry, MessageEntry) and entry.metadata:
                if external_id in (
                    entry.metadata.get("external_id"),
                    entry.metadata.get("bot_response_id"),
                ):
                    return entry
        return None

    async def get_messages_around(
        self, message_id: str, window: int = 3
    ) -> list[MessageEntry]:
        messages = [e for e in await self.load_entries() if isinstance(e, MessageEntry)]

        target_idx = next(
            (i for i, msg in enumerate(messages) if msg.id == message_id), None
        )
        # Fallback: search by external_id or bot_response_id in metadata
        if target_idx is None:
            target_idx = next(
                (
                    i
                    for i, msg in enumerate(messages)
                    if msg.metadata
                    and message_id
                    in (
                        msg.metadata.get("external_id"),
                        msg.metadata.get("bot_response_id"),
                    )
                ),
                None,
            )
        if target_idx is None:
            return []

        start = max(0, target_idx - window)
        end = min(len(messages), target_idx + window + 1)
        return messages[start:end]

    async def search_messages(self, query: str, limit: int = 20) -> list[MessageEntry]:
        messages = [e for e in await self.load_entries() if isinstance(e, MessageEntry)]
        query_lower = query.lower()

        results: list[MessageEntry] = []
        for msg in reversed(messages):
            if query_lower in msg._extract_text_content().lower():
                results.append(msg)
                if len(results) >= limit:
                    break
        return results

    async def load_messages_for_branch(
        self,
        head_message_id: str,
        branch_id: str | None = None,
        token_budget: int | None = None,
        recency_window: int = DEFAULT_RECENCY_WINDOW,
        include_timestamps: bool = False,
    ) -> tuple[list[Message], list[str]]:
        """Load messages for a specific branch, following the parent_id chain."""
        entries = await self.load_entries()
        branch_entries = self._resolve_branch(entries, head_message_id, branch_id)
        messages, message_ids, token_counts = self._build_messages(
            branch_entries, include_timestamps=include_timestamps
        )
        return prune_messages_to_budget(
            messages,
            token_counts,
            token_budget,
            recency_window,
            message_ids,
        )

    async def load_subagent_entries(self, agent_session_id: str) -> list[Entry]:
        """Load entries from a subagent JSONL file."""
        path = self.session_dir / "subagents" / f"{agent_session_id}.jsonl"
        if not path.exists():
            return []

        entries: list[Entry] = []
        async with aiofiles.open(path, encoding="utf-8") as f:
            line_num = 0
            async for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(parse_entry(json.loads(line)))
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        "subagent_parse_error",
                        extra={
                            "line_num": line_num,
                            "file": str(path),
                            "error.message": str(e),
                        },
                    )
        return entries

    def build_subagent_messages(
        self, entries: list[Entry]
    ) -> tuple[list[Message], list[str], list[int]]:
        """Build LLM messages from subagent entries.

        Unlike _build_messages(), this includes entries with agent_session_id
        set since all subagent entries have one.
        """
        from ash.core.tokens import estimate_message_tokens

        tool_use_ids: set[str] = set()
        for entry in entries:
            if isinstance(entry, ToolUseEntry):
                tool_use_ids.add(entry.id)
            elif isinstance(entry, MessageEntry) and isinstance(entry.content, list):
                for block in entry.content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_use_ids.add(block["id"])

        messages: list[Message] = []
        message_ids: list[str] = []
        token_counts: list[int] = []
        pending_results: list[ToolResult] = []

        def flush_pending_results() -> None:
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
            if isinstance(entry, (SessionHeader, ToolUseEntry, CompactionEntry)):
                continue

            if isinstance(entry, MessageEntry):
                flush_pending_results()
                content = self._convert_content(entry.content)
                if not content:
                    continue
                messages.append(Message(role=Role(entry.role), content=content))
                message_ids.append(entry.id)
                token_counts.append(
                    entry.token_count
                    if entry.token_count is not None
                    else estimate_message_tokens(entry.role, entry.content)
                )

            elif (
                isinstance(entry, ToolResultEntry) and entry.tool_use_id in tool_use_ids
            ):
                pending_results.append(
                    ToolResult(
                        tool_use_id=entry.tool_use_id,
                        content=entry.output,
                        is_error=not entry.success,
                    )
                )

        flush_pending_results()
        return messages, message_ids, token_counts

    def _resolve_branch(
        self,
        entries: list[Entry],
        head_message_id: str,
        branch_id: str | None = None,
    ) -> list[Entry]:
        """Filter entries to only those on the branch ending at head_message_id.

        Walks the parent_id chain from head back to root to determine which
        messages are on the branch. Then filters tool_use, tool_result,
        agent_session, agent_session_complete, and compaction entries to only
        those associated with branch messages.

        For v1 compatibility: messages with parent_id=None are treated as an
        implicit linear chain in file order.
        """
        from ash.sessions.types import AgentSessionCompleteEntry, AgentSessionEntry

        # Index messages by id, preserving file order
        msg_by_id: dict[str, MessageEntry] = {}
        msg_order: list[str] = []
        for entry in entries:
            if isinstance(entry, MessageEntry):
                msg_by_id[entry.id] = entry
                msg_order.append(entry.id)

        if head_message_id not in msg_by_id:
            return list(entries)  # Fallback: return all if head not found

        # Walk parent_id chain from head to root
        branch_msg_ids: set[str] = set()
        current_id: str | None = head_message_id
        while current_id is not None:
            if current_id in branch_msg_ids:
                break  # Cycle protection
            branch_msg_ids.add(current_id)
            msg = msg_by_id.get(current_id)
            if msg is None:
                break
            if msg.parent_id is not None:
                current_id = msg.parent_id
            else:
                # v1 fallback: include all preceding messages in file order
                idx = msg_order.index(current_id) if current_id in msg_order else 0
                for prev_id in msg_order[:idx]:
                    branch_msg_ids.add(prev_id)
                break

        # Collect tool_use IDs that belong to branch messages
        branch_tool_use_ids: set[str] = set()
        for entry in entries:
            if isinstance(entry, ToolUseEntry) and entry.message_id in branch_msg_ids:
                branch_tool_use_ids.add(entry.id)
            elif isinstance(entry, MessageEntry) and entry.id in branch_msg_ids:
                if isinstance(entry.content, list):
                    for block in entry.content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            branch_tool_use_ids.add(block["id"])

        # Collect agent_session IDs that belong to branch tool uses
        branch_agent_session_ids: set[str] = set()
        for entry in entries:
            if (
                isinstance(entry, AgentSessionEntry)
                and entry.parent_tool_use_id in branch_tool_use_ids
            ):
                branch_agent_session_ids.add(entry.id)

        # Filter entries to branch membership
        result: list[Entry] = []
        for entry in entries:
            if isinstance(entry, SessionHeader):
                result.append(entry)
            elif isinstance(entry, MessageEntry):
                if entry.id in branch_msg_ids:
                    result.append(entry)
            elif isinstance(entry, ToolUseEntry):
                if entry.id in branch_tool_use_ids:
                    result.append(entry)
            elif isinstance(entry, ToolResultEntry):
                if entry.tool_use_id in branch_tool_use_ids:
                    result.append(entry)
            elif isinstance(entry, AgentSessionEntry):
                if entry.id in branch_agent_session_ids:
                    result.append(entry)
            elif isinstance(entry, AgentSessionCompleteEntry):
                if entry.agent_session_id in branch_agent_session_ids:
                    result.append(entry)
            elif isinstance(entry, CompactionEntry):
                # Include compaction if it's for this branch or unscoped (None)
                if entry.branch_id is None or entry.branch_id == branch_id:
                    result.append(entry)

        return result
