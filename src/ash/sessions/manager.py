"""Session manager for orchestrating JSONL session operations."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ash.config.paths import get_ash_home
from ash.llm.types import ContentBlock, Message, ToolUse
from ash.sessions.reader import SessionReader
from ash.sessions.types import (
    CompactionEntry,
    MessageEntry,
    SessionHeader,
    SessionState,
    ToolResultEntry,
    ToolUseEntry,
    session_key,
)
from ash.sessions.utils import content_block_to_dict
from ash.sessions.writer import SessionWriter

logger = logging.getLogger(__name__)

STATE_FILENAME = "state.json"


def get_sessions_path() -> Path:
    return get_ash_home() / "sessions"


class SessionManager:
    def __init__(
        self,
        provider: str,
        chat_id: str | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
        sessions_path: Path | None = None,
    ) -> None:
        self.provider = provider
        self.chat_id = chat_id
        self.user_id = user_id
        self.thread_id = thread_id
        self._key = session_key(provider, chat_id, user_id, thread_id)
        self._session_dir = (sessions_path or get_sessions_path()) / self._key
        self._reader = SessionReader(self._session_dir)
        self._writer = SessionWriter(self._session_dir)
        self._header: SessionHeader | None = None
        self._current_message_id: str | None = None

    @property
    def session_key(self) -> str:
        return self._key

    @property
    def session_dir(self) -> Path:
        return self._session_dir

    @property
    def session_id(self) -> str:
        return self._header.id if self._header else ""

    @property
    def state_path(self) -> Path:
        return self._session_dir / STATE_FILENAME

    def exists(self) -> bool:
        return self._reader.exists()

    async def ensure_session(self) -> SessionHeader:
        if self._header is not None:
            return self._header

        self._header = await self._reader.load_header()
        if self._header is not None:
            self._ensure_state_file()
            return self._header

        self._header = SessionHeader.create(
            provider=self.provider,
            user_id=self.user_id,
            chat_id=self.chat_id,
        )
        await self._writer.write_header(self._header)
        self._ensure_state_file()
        logger.info("Created new session: %s", self._key)
        return self._header

    def _ensure_state_file(self) -> None:
        if self.state_path.exists():
            return

        state = SessionState(
            provider=self.provider,
            chat_id=self.chat_id,
            user_id=self.user_id,
            thread_id=self.thread_id,
        )
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state.model_dump(mode="json"), indent=2, default=str)
        )

    async def add_user_message(
        self,
        content: str,
        token_count: int | None = None,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
        username: str | None = None,
        display_name: str | None = None,
    ) -> str:
        await self.ensure_session()
        entry = MessageEntry.create(
            role="user",
            content=content,
            token_count=token_count,
            user_id=user_id or self.user_id,
            username=username,
            display_name=display_name,
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
        await self.ensure_session()
        stored_content: str | list[dict[str, Any]] = (
            content
            if isinstance(content, str)
            else [content_block_to_dict(b) for b in content]
        )
        entry = MessageEntry.create(
            role="assistant",
            content=stored_content,
            token_count=token_count,
            metadata=metadata,
        )
        await self._writer.write_message(entry)
        self._current_message_id = entry.id

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

    async def add_tool_use(
        self,
        tool_use_id: str,
        name: str,
        input_data: dict[str, Any],
    ) -> None:
        await self.ensure_session()
        entry = ToolUseEntry.create(
            tool_use_id=tool_use_id,
            message_id=self._current_message_id or "",
            name=name,
            input_data=input_data,
        )
        await self._writer.write_tool_use(entry)

    async def add_tool_result(
        self,
        tool_use_id: str,
        output: str,
        success: bool = True,
        duration_ms: int | None = None,
    ) -> None:
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
        return await self._reader.load_messages_for_llm(
            token_budget, recency_window, include_timestamps
        )

    async def get_message_ids(self) -> set[str]:
        return await self._reader.get_message_ids()

    async def get_recent_message_ids(self, recency_window: int = 10) -> set[str]:
        all_ids = list(await self._reader.get_message_ids())
        if not all_ids:
            return set()
        return set(all_ids[max(0, len(all_ids) - recency_window) :])

    async def has_message_with_external_id(self, external_id: str) -> bool:
        return await self._reader.has_message_with_external_id(external_id)

    async def get_message_by_external_id(self, external_id: str) -> MessageEntry | None:
        return await self._reader.get_message_by_external_id(external_id)

    async def get_messages_around(
        self, message_id: str, window: int = 3
    ) -> list[MessageEntry]:
        return await self._reader.get_messages_around(message_id, window)

    async def search_messages(self, query: str, limit: int = 20) -> list[MessageEntry]:
        return await self._reader.search_messages(query, limit)

    async def get_last_message_time(self) -> datetime | None:
        entries = await self._reader.load_entries()
        msg_entries = [e for e in entries if isinstance(e, MessageEntry)]
        return msg_entries[-1].created_at if msg_entries else None

    @classmethod
    async def list_sessions(
        cls, sessions_path: Path | None = None
    ) -> list[dict[str, Any]]:
        base_path = sessions_path or get_sessions_path()
        if not base_path.exists():
            return []

        sessions = []
        for session_dir in sorted(base_path.iterdir()):
            if not session_dir.is_dir():
                continue
            header = await SessionReader(session_dir).load_header()
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
        base_path = sessions_path or get_sessions_path()
        session_dir = base_path / key

        if not session_dir.exists():
            return None

        header = await SessionReader(session_dir).load_header()
        if not header:
            return None

        return cls(
            provider=header.provider,
            chat_id=header.chat_id,
            user_id=header.user_id,
            sessions_path=base_path,
        )
