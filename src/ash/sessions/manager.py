"""Session manager for orchestrating JSONL session operations."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ash.config.paths import get_ash_home
from ash.llm.types import ContentBlock, Message, ToolUse
from ash.sessions.reader import SessionReader
from ash.sessions.types import (
    AgentSessionEntry,
    BranchHead,
    CompactionEntry,
    MessageEntry,
    SessionHeader,
    SessionState,
    ToolResultEntry,
    ToolUseEntry,
    generate_id,
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
        agent_session_id: str | None = None,
        parent_id: str | None = None,
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
            agent_session_id=agent_session_id,
            parent_id=parent_id or self._current_message_id,
        )
        await self._writer.write_message(entry)
        self._current_message_id = entry.id
        return entry.id

    async def add_assistant_message(
        self,
        content: str | list[ContentBlock] | list[dict[str, Any]],
        token_count: int | None = None,
        metadata: dict[str, Any] | None = None,
        agent_session_id: str | None = None,
        parent_id: str | None = None,
    ) -> str:
        await self.ensure_session()
        stored_content: str | list[dict[str, Any]]
        if isinstance(content, str):
            stored_content = content
        elif content and isinstance(content[0], dict):
            # Already serialized dicts (from subagent logging)
            stored_content = content  # type: ignore[assignment]
        else:
            # ContentBlock objects need serialization
            stored_content = [content_block_to_dict(b) for b in content]  # type: ignore[arg-type]

        entry = MessageEntry.create(
            role="assistant",
            content=stored_content,
            token_count=token_count,
            metadata=metadata,
            agent_session_id=agent_session_id,
            parent_id=parent_id or self._current_message_id,
        )
        await self._writer.write_message(entry)
        self._current_message_id = entry.id

        # Auto-extract tool uses from ContentBlock content
        if (
            not isinstance(content, str)
            and content
            and not isinstance(content[0], dict)
        ):
            for block in content:
                if isinstance(block, ToolUse):
                    tool_entry = ToolUseEntry.create(
                        tool_use_id=block.id,
                        message_id=entry.id,
                        name=block.name,
                        input_data=block.input,
                        agent_session_id=agent_session_id,
                    )
                    await self._writer.write_tool_use(tool_entry)

        return entry.id

    async def add_tool_use(
        self,
        tool_use_id: str,
        name: str,
        input_data: dict[str, Any],
        agent_session_id: str | None = None,
    ) -> None:
        await self.ensure_session()
        entry = ToolUseEntry.create(
            tool_use_id=tool_use_id,
            message_id=self._current_message_id or "",
            name=name,
            input_data=input_data,
            agent_session_id=agent_session_id,
        )
        await self._writer.write_tool_use(entry)

    async def add_tool_result(
        self,
        tool_use_id: str,
        output: str,
        success: bool = True,
        duration_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
        agent_session_id: str | None = None,
    ) -> None:
        await self.ensure_session()
        entry = ToolResultEntry.create(
            tool_use_id=tool_use_id,
            output=output,
            success=success,
            duration_ms=duration_ms,
            metadata=metadata,
            agent_session_id=agent_session_id,
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

    async def start_agent_session(
        self,
        parent_tool_use_id: str,
        agent_type: Literal["skill", "agent"],
        agent_name: str,
    ) -> str:
        """Start a new subagent session and log it."""
        await self.ensure_session()
        entry = AgentSessionEntry.create(
            parent_tool_use_id=parent_tool_use_id,
            agent_type=agent_type,
            agent_name=agent_name,
        )
        await self._writer.write_agent_session(entry)
        return entry.id

    async def load_messages_for_llm(
        self,
        token_budget: int | None = None,
        recency_window: int = 10,
        include_timestamps: bool = False,
        branch_head_id: str | None = None,
        branch_id: str | None = None,
    ) -> tuple[list[Message], list[str]]:
        if branch_head_id is not None:
            return await self._reader.load_messages_for_branch(
                branch_head_id,
                branch_id=branch_id,
                token_budget=token_budget,
                recency_window=recency_window,
                include_timestamps=include_timestamps,
            )
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

    def fork_at_message(self, message_id: str) -> str:
        """Create a new branch forking from the given message.

        If no branches exist yet, creates a "main" branch for the current tip.
        Returns the new branch_id.
        """
        state = self._load_state()
        if state is None:
            state = SessionState(provider=self.provider)

        # If no branches tracked yet, create "main" for the current linear tip
        if not state.branches and self._current_message_id:
            main_branch = BranchHead(
                branch_id=generate_id(),
                head_message_id=self._current_message_id,
                fork_point_id=None,
            )
            state.branches.append(main_branch)

        # Create new branch forking from message_id
        branch_id = generate_id()
        new_branch = BranchHead(
            branch_id=branch_id,
            head_message_id=message_id,
            fork_point_id=message_id,
        )
        state.branches.append(new_branch)

        # Set current position to fork point so next write chains from there
        self._current_message_id = message_id

        self._save_state(state)
        return branch_id

    def update_branch_head(self, branch_id: str, head_message_id: str) -> None:
        """Update the head of a branch after writing a message."""
        state = self._load_state()
        if state is None:
            return

        for branch in state.branches:
            if branch.branch_id == branch_id:
                branch.head_message_id = head_message_id
                self._save_state(state)
                return

    def get_branch_for_message(self, message_id: str) -> BranchHead | None:
        """Find the branch that has message_id as its head."""

        state = self._load_state()
        if state is None:
            return None

        for branch in state.branches:
            if branch.head_message_id == message_id:
                return branch
        return None

    def _load_state(self) -> SessionState | None:
        """Load the session state from state.json."""
        if not self.state_path.exists():
            return None
        try:
            data = json.loads(self.state_path.read_text())
            return SessionState.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to load session state: {e}")
            return None

    def _save_state(self, state: SessionState) -> None:
        """Save the session state to state.json."""
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state.model_dump(mode="json"), indent=2, default=str)
        )

    async def get_pending_checkpoint_from_log(
        self, truncated_id: str
    ) -> tuple[ToolUseEntry, ToolResultEntry, dict[str, Any]] | None:
        """Find a pending checkpoint by truncated ID from session log.

        Returns (tool_use, tool_result, checkpoint_dict) or None.
        """
        entries = await self._reader.load_entries()

        # Build tool_use lookup
        tool_uses: dict[str, ToolUseEntry] = {}
        for entry in entries:
            if isinstance(entry, ToolUseEntry):
                tool_uses[entry.id] = entry

        # Find matching checkpoint in tool results (reverse order = most recent)
        for entry in reversed(entries):
            if not isinstance(entry, ToolResultEntry):
                continue
            if not entry.metadata or "checkpoint" not in entry.metadata:
                continue

            checkpoint = entry.metadata["checkpoint"]
            if checkpoint.get("checkpoint_id", "")[:55] == truncated_id:
                tool_use = tool_uses.get(entry.tool_use_id)
                if tool_use:
                    return (tool_use, entry, checkpoint)

        return None

    async def has_bot_response_id(self, bot_response_id: str) -> bool:
        """Check if this session contains a message with the given bot_response_id."""
        entries = await self._reader.load_entries()
        for entry in entries:
            if isinstance(entry, MessageEntry):
                if (
                    entry.metadata
                    and entry.metadata.get("bot_response_id") == bot_response_id
                ):
                    return True
        return False
