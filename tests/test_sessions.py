"""Tests for JSONL session management."""

from __future__ import annotations

import json

import pytest

from ash.llm.types import TextContent, ToolUse
from ash.sessions.manager import SessionManager
from ash.sessions.reader import SessionReader
from ash.sessions.types import (
    parse_entry,
    session_key,
)
from ash.sessions.writer import SessionWriter


class TestSessionKey:
    """Tests for session key generation - core scoping logic."""

    def test_provider_only(self):
        assert session_key("cli") == "cli"

    def test_provider_with_chat_id(self):
        assert session_key("telegram", chat_id="12345") == "telegram_12345"

    def test_provider_with_user_id(self):
        assert session_key("api", user_id="user123") == "api_user123"

    def test_chat_id_takes_precedence_over_user_id(self):
        """Chat-level sessions override user-level."""
        assert session_key("api", chat_id="chat1", user_id="user1") == "api_chat1"

    def test_thread_id_creates_subsession(self):
        """Thread ID creates sub-session within a chat."""
        assert (
            session_key("telegram", chat_id="123", thread_id="42") == "telegram_123_42"
        )

    def test_sanitizes_special_characters(self):
        """Prevents path traversal and invalid filesystem chars."""
        assert session_key("cli", chat_id="test@user.com") == "cli_test_user_com"

    def test_limits_length(self):
        """Prevents overly long directory names."""
        long_id = "a" * 100
        key = session_key("cli", chat_id=long_id)
        assert len(key) <= 68  # provider + _ + max 64 chars


class TestParseEntry:
    """Tests for entry parsing error handling."""

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown entry type"):
            parse_entry({"type": "unknown"})


class TestSessionWriter:
    """Integration tests for SessionWriter."""

    @pytest.fixture
    def session_dir(self, tmp_path):
        return tmp_path / "test_session"

    @pytest.fixture
    def writer(self, session_dir):
        return SessionWriter(session_dir)

    @pytest.mark.asyncio
    async def test_writes_to_correct_files(self, writer, session_dir):
        """Messages go to both context.jsonl and history.jsonl."""
        from ash.sessions.types import MessageEntry, SessionHeader

        await writer.write_header(SessionHeader.create(provider="cli"))
        await writer.write_message(MessageEntry.create(role="user", content="Hello!"))

        # Context has full message with type
        context = json.loads((session_dir / "context.jsonl").read_text().split("\n")[1])
        assert context["type"] == "message"
        assert context["content"] == "Hello!"

        # History has simplified format without type
        history = json.loads((session_dir / "history.jsonl").read_text().strip())
        assert "type" not in history
        assert history["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_tool_entries_context_only(self, writer, session_dir):
        """Tool use/results only go to context, not history."""
        from ash.sessions.types import ToolResultEntry, ToolUseEntry

        await writer.write_tool_use(
            ToolUseEntry.create(
                tool_use_id="t1", message_id="m1", name="bash", input_data={}
            )
        )
        await writer.write_tool_result(
            ToolResultEntry.create(tool_use_id="t1", output="ok", success=True)
        )

        assert (session_dir / "context.jsonl").exists()
        assert not (session_dir / "history.jsonl").exists()


class TestSessionReader:
    """Integration tests for SessionReader."""

    @pytest.fixture
    def session_dir(self, tmp_path):
        return tmp_path / "test_session"

    @pytest.fixture
    def reader(self, session_dir):
        return SessionReader(session_dir)

    @pytest.mark.asyncio
    async def test_load_entries_parses_all_types(self, reader, session_dir):
        """Reader correctly parses all entry types from JSONL."""
        from ash.sessions.types import (
            MessageEntry,
            SessionHeader,
            ToolResultEntry,
            ToolUseEntry,
        )

        session_dir.mkdir(parents=True)
        lines = [
            '{"type":"session","version":"1","id":"s1","created_at":"2026-01-11T10:00:00+00:00","provider":"cli"}',
            '{"type":"message","id":"m1","role":"user","content":"Hello","created_at":"2026-01-11T10:00:01+00:00"}',
            '{"type":"tool_use","id":"t1","message_id":"m1","name":"bash","input":{}}',
            '{"type":"tool_result","tool_use_id":"t1","output":"result","success":true}',
        ]
        (session_dir / "context.jsonl").write_text("\n".join(lines) + "\n")

        entries = await reader.load_entries()

        assert len(entries) == 4
        assert isinstance(entries[0], SessionHeader)
        assert isinstance(entries[1], MessageEntry)
        assert isinstance(entries[2], ToolUseEntry)
        assert isinstance(entries[3], ToolResultEntry)

    @pytest.mark.asyncio
    async def test_load_messages_for_llm(self, reader, session_dir):
        """Converts stored messages to LLM-ready format."""
        session_dir.mkdir(parents=True)
        lines = [
            '{"type":"session","version":"1","id":"s1","created_at":"2026-01-11T10:00:00+00:00","provider":"cli"}',
            '{"type":"message","id":"m1","role":"user","content":"Hello","created_at":"2026-01-11T10:00:01+00:00"}',
            '{"type":"message","id":"m2","role":"assistant","content":"Hi!","created_at":"2026-01-11T10:00:02+00:00"}',
        ]
        (session_dir / "context.jsonl").write_text("\n".join(lines) + "\n")

        messages, ids = await reader.load_messages_for_llm()

        assert len(messages) == 2
        assert messages[0].role.value == "user"
        assert messages[0].content == "Hello"
        assert ids == ["m1", "m2"]


class TestSessionManager:
    """Integration tests for SessionManager."""

    @pytest.fixture
    def sessions_path(self, tmp_path):
        return tmp_path / "sessions"

    @pytest.fixture
    def manager(self, sessions_path):
        return SessionManager(provider="cli", sessions_path=sessions_path)

    @pytest.mark.asyncio
    async def test_full_conversation_lifecycle(self, manager):
        """Complete conversation with tool use roundtrips correctly."""
        await manager.ensure_session()

        # User message
        await manager.add_user_message("List files")

        # Assistant with tool use
        await manager.add_assistant_message(
            [
                TextContent(text="Let me check."),
                ToolUse(id="t1", name="bash", input={"command": "ls"}),
            ]
        )

        # Tool result
        await manager.add_tool_result(
            tool_use_id="t1", output="file1.txt\nfile2.txt", success=True
        )

        # Final response
        await manager.add_assistant_message("Found 2 files.")

        # Verify roundtrip
        messages, _ = await manager.load_messages_for_llm()
        assert len(messages) >= 3

    @pytest.mark.asyncio
    async def test_list_and_get_sessions(self, sessions_path):
        """Can list all sessions and retrieve by key."""
        m1 = SessionManager(provider="cli", sessions_path=sessions_path)
        await m1.ensure_session()

        m2 = SessionManager(
            provider="telegram", chat_id="123", sessions_path=sessions_path
        )
        await m2.ensure_session()
        await m2.add_user_message("Test")

        # List shows both
        sessions = await SessionManager.list_sessions(sessions_path)
        assert len(sessions) == 2
        assert {s["provider"] for s in sessions} == {"cli", "telegram"}

        # Can retrieve by key
        retrieved = await SessionManager.get_session("telegram_123", sessions_path)
        assert retrieved is not None
        messages, _ = await retrieved.load_messages_for_llm()
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, sessions_path):
        result = await SessionManager.get_session("nonexistent", sessions_path)
        assert result is None
