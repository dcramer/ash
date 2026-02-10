"""JSONL session writer for dual-file output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles

if TYPE_CHECKING:
    from ash.sessions.types import (
        AgentSessionEntry,
        CompactionEntry,
        MessageEntry,
        SessionHeader,
        ToolResultEntry,
        ToolUseEntry,
    )


class SessionWriter:
    """Writes session entries to JSONL files.

    Maintains two files:
    - context.jsonl: Full LLM context (all entry types)
    - history.jsonl: Human-readable conversation (messages only)
    """

    def __init__(self, session_dir: Path) -> None:
        """Initialize session writer.

        Args:
            session_dir: Directory to write session files to.
        """
        self.session_dir = session_dir
        self.context_file = session_dir / "context.jsonl"
        self.history_file = session_dir / "history.jsonl"
        self._initialized = False

    async def ensure_directory(self) -> None:
        """Ensure the session directory exists."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    async def write_header(self, header: SessionHeader) -> None:
        """Write session header (first entry in context.jsonl).

        Args:
            header: Session header entry.
        """
        if not self._initialized:
            await self.ensure_directory()
        await self._append_context(header.to_dict())

    async def write_message(self, entry: MessageEntry) -> None:
        """Write a message entry to both files.

        Args:
            entry: Message entry to write.
        """
        if not self._initialized:
            await self.ensure_directory()
        # Write to context.jsonl (full format)
        await self._append_context(entry.to_dict())
        # Write to history.jsonl (simplified format)
        await self._append_history(entry.to_history_dict())

    async def _write_context_entry(
        self,
        entry: ToolUseEntry | ToolResultEntry | CompactionEntry | AgentSessionEntry,
    ) -> None:
        """Write an entry to context.jsonl only."""
        if not self._initialized:
            await self.ensure_directory()
        await self._append_context(entry.to_dict())

    async def write_tool_use(self, entry: ToolUseEntry) -> None:
        """Write a tool use entry to context.jsonl only."""
        await self._write_context_entry(entry)

    async def write_tool_result(self, entry: ToolResultEntry) -> None:
        """Write a tool result entry to context.jsonl only."""
        await self._write_context_entry(entry)

    async def write_compaction(self, entry: CompactionEntry) -> None:
        """Write a compaction entry to context.jsonl only."""
        await self._write_context_entry(entry)

    async def write_agent_session(self, entry: AgentSessionEntry) -> None:
        """Write an agent session entry to context.jsonl only."""
        await self._write_context_entry(entry)

    async def _append_context(self, data: dict) -> None:
        """Append a JSON line to context.jsonl.

        Args:
            data: Data to write as JSON.
        """
        line = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        async with aiofiles.open(self.context_file, "a", encoding="utf-8") as f:
            await f.write(line + "\n")

    async def _append_history(self, data: dict) -> None:
        """Append a JSON line to history.jsonl.

        Args:
            data: Data to write as JSON.
        """
        line = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        async with aiofiles.open(self.history_file, "a", encoding="utf-8") as f:
            await f.write(line + "\n")

    def exists(self) -> bool:
        """Check if session files exist.

        Returns:
            True if context.jsonl exists.
        """
        return self.context_file.exists()
