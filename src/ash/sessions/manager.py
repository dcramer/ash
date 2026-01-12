"""Session manager for orchestrating JSONL session operations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ash.config.paths import get_ash_home
from ash.llm.types import ContentBlock, Message, ToolUse
from ash.sessions.reader import SessionReader
from ash.sessions.types import (
    CompactionEntry,
    MessageEntry,
    SessionHeader,
    ToolResultEntry,
    ToolUseEntry,
    session_key,
)
from ash.sessions.utils import content_block_to_dict
from ash.sessions.writer import SessionWriter

logger = logging.getLogger(__name__)


def get_sessions_path() -> Path:
    """Get the sessions directory path.

    Returns:
        Path to ~/.ash/sessions/
    """
    return get_ash_home() / "sessions"


class SessionManager:
    """Manages session lifecycle and persistence.

    Provides a high-level interface for:
    - Creating and loading sessions
    - Writing messages and tool interactions
    - Loading context for LLM
    - Session identification via composite keys
    """

    def __init__(
        self,
        provider: str,
        chat_id: str | None = None,
        user_id: str | None = None,
        sessions_path: Path | None = None,
    ) -> None:
        """Initialize session manager.

        Args:
            provider: Provider name (e.g., "cli", "telegram", "api").
            chat_id: Optional chat/conversation ID.
            user_id: Optional user ID.
            sessions_path: Override sessions directory (for testing).
        """
        self.provider = provider
        self.chat_id = chat_id
        self.user_id = user_id

        # Compute session key and path
        self._key = session_key(provider, chat_id, user_id)
        base_path = sessions_path or get_sessions_path()
        self._session_dir = base_path / self._key

        # Initialize reader and writer
        self._reader = SessionReader(self._session_dir)
        self._writer = SessionWriter(self._session_dir)

        # Cached header
        self._header: SessionHeader | None = None

        # Track current message ID for linking tool uses
        self._current_message_id: str | None = None

    @property
    def session_key(self) -> str:
        """Get the session key (directory name)."""
        return self._key

    @property
    def session_dir(self) -> Path:
        """Get the session directory path."""
        return self._session_dir

    @property
    def session_id(self) -> str:
        """Get the session ID (from header).

        Note: Call ensure_session() first to populate the header.
        """
        if self._header is None:
            return ""
        return self._header.id

    def exists(self) -> bool:
        """Check if session already exists.

        Returns:
            True if session files exist.
        """
        return self._reader.exists()

    async def ensure_session(self) -> SessionHeader:
        """Ensure session exists, creating if needed.

        Returns:
            Session header.
        """
        if self._header is not None:
            return self._header

        # Try to load existing
        self._header = await self._reader.load_header()
        if self._header is not None:
            return self._header

        # Create new session
        self._header = SessionHeader.create(
            provider=self.provider,
            user_id=self.user_id,
            chat_id=self.chat_id,
        )
        await self._writer.write_header(self._header)
        logger.info("Created new session: %s", self._key)

        return self._header

    async def add_user_message(
        self,
        content: str,
        token_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a user message to the session.

        Args:
            content: Message content.
            token_count: Optional pre-computed token count.
            metadata: Optional metadata (e.g., external_id for deduplication).

        Returns:
            Message ID.
        """
        await self.ensure_session()

        entry = MessageEntry.create(
            role="user",
            content=content,
            token_count=token_count,
            user_id=self.user_id,
            metadata=metadata,
        )
        await self._writer.write_message(entry)
        self._current_message_id = entry.id

        return entry.id

    async def add_assistant_message(
        self,
        content: str | list[ContentBlock],
        token_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add an assistant message to the session.

        Args:
            content: Message content (string or content blocks).
            token_count: Optional pre-computed token count.
            metadata: Optional metadata (e.g., bot_response_id).

        Returns:
            Message ID.
        """
        await self.ensure_session()

        # Convert ContentBlock objects to dicts for storage
        stored_content: str | list[dict[str, Any]]
        if isinstance(content, str):
            stored_content = content
        else:
            stored_content = [content_block_to_dict(b) for b in content]

        entry = MessageEntry.create(
            role="assistant",
            content=stored_content,
            token_count=token_count,
            metadata=metadata,
        )
        await self._writer.write_message(entry)
        self._current_message_id = entry.id

        # Also write tool uses as separate entries (for logging/debugging)
        if not isinstance(content, str):
            for block in content:
                if isinstance(block, ToolUse):
                    tool_entry = ToolUseEntry.create(
                        tool_use_id=block.id,
                        message_id=entry.id,
                        name=block.name,
                        input_data=block.input,
                    )
                    await self._writer.write_tool_use(tool_entry)

        return entry.id

    async def add_tool_result(
        self,
        tool_use_id: str,
        output: str,
        success: bool = True,
        duration_ms: int | None = None,
    ) -> None:
        """Add a tool result to the session.

        Note: Tool results are stored separately but will be combined
        with the next user turn when loading for LLM.

        Args:
            tool_use_id: ID of the tool use this is a result for.
            output: Tool output content.
            success: Whether the tool executed successfully.
            duration_ms: Optional execution duration.
        """
        await self.ensure_session()

        entry = ToolResultEntry.create(
            tool_use_id=tool_use_id,
            output=output,
            success=success,
            duration_ms=duration_ms,
        )
        await self._writer.write_tool_result(entry)

    async def add_compaction(
        self,
        summary: str,
        tokens_before: int,
        tokens_after: int,
        first_kept_entry_id: str,
    ) -> None:
        """Record a compaction event.

        Args:
            summary: Summary of compacted content.
            tokens_before: Token count before compaction.
            tokens_after: Token count after compaction.
            first_kept_entry_id: ID of first entry kept after compaction.
        """
        await self.ensure_session()

        entry = CompactionEntry.create(
            summary=summary,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            first_kept_entry_id=first_kept_entry_id,
        )
        await self._writer.write_compaction(entry)

    async def load_messages_for_llm(
        self,
        token_budget: int | None = None,
        recency_window: int = 10,
        include_timestamps: bool = False,
    ) -> tuple[list[Message], list[str]]:
        """Load messages formatted for LLM API.

        Args:
            token_budget: Maximum tokens for messages.
            recency_window: Always keep at least this many recent messages.
            include_timestamps: Whether to prefix messages with timestamps.

        Returns:
            Tuple of (messages, message_ids).
        """
        return await self._reader.load_messages_for_llm(
            token_budget, recency_window, include_timestamps
        )

    async def get_message_ids(self) -> set[str]:
        """Get all message IDs in the session.

        Returns:
            Set of message IDs.
        """
        return await self._reader.get_message_ids()

    async def get_recent_message_ids(self, recency_window: int = 10) -> set[str]:
        """Get message IDs in the recency window.

        Args:
            recency_window: Number of recent messages.

        Returns:
            Set of message IDs.
        """
        all_ids = list(await self._reader.get_message_ids())
        if not all_ids:
            return set()
        start = max(0, len(all_ids) - recency_window)
        return set(all_ids[start:])

    async def has_message_with_external_id(self, external_id: str) -> bool:
        """Check if a message with given external ID exists.

        Args:
            external_id: External message ID.

        Returns:
            True if message exists.
        """
        return await self._reader.has_message_with_external_id(external_id)

    async def get_message_by_external_id(self, external_id: str) -> MessageEntry | None:
        """Find message by external ID.

        Args:
            external_id: External message ID.

        Returns:
            MessageEntry if found, None otherwise.
        """
        return await self._reader.get_message_by_external_id(external_id)

    async def get_messages_around(
        self, message_id: str, window: int = 3
    ) -> list[MessageEntry]:
        """Get messages around a specific message.

        Args:
            message_id: Target message ID.
            window: Number of messages before and after.

        Returns:
            List of MessageEntry sorted by created_at.
        """
        return await self._reader.get_messages_around(message_id, window)

    async def search_messages(self, query: str, limit: int = 20) -> list[MessageEntry]:
        """Search messages by content.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            List of matching MessageEntry.
        """
        return await self._reader.search_messages(query, limit)

    @classmethod
    async def list_sessions(
        cls, sessions_path: Path | None = None
    ) -> list[dict[str, Any]]:
        """List all sessions.

        Args:
            sessions_path: Override sessions directory.

        Returns:
            List of session info dicts with keys: key, provider, chat_id, user_id, created_at.
        """
        base_path = sessions_path or get_sessions_path()
        if not base_path.exists():
            return []

        sessions = []
        for session_dir in sorted(base_path.iterdir()):
            if not session_dir.is_dir():
                continue

            reader = SessionReader(session_dir)
            header = await reader.load_header()
            if header:
                sessions.append(
                    {
                        "key": session_dir.name,
                        "id": header.id,
                        "provider": header.provider,
                        "chat_id": header.chat_id,
                        "user_id": header.user_id,
                        "created_at": header.created_at,
                    }
                )

        return sessions

    @classmethod
    async def get_session(
        cls,
        key: str,
        sessions_path: Path | None = None,
    ) -> SessionManager | None:
        """Get a session manager by key.

        Args:
            key: Session key (directory name).
            sessions_path: Override sessions directory.

        Returns:
            SessionManager or None if not found.
        """
        base_path = sessions_path or get_sessions_path()
        session_dir = base_path / key

        if not session_dir.exists():
            return None

        reader = SessionReader(session_dir)
        header = await reader.load_header()
        if not header:
            return None

        return cls(
            provider=header.provider,
            chat_id=header.chat_id,
            user_id=header.user_id,
            sessions_path=base_path,
        )
