"""Tests for JSONL session management."""

from __future__ import annotations

import json

import pytest

from ash.llm.types import TextContent, ToolUse
from ash.sessions.manager import SessionManager
from ash.sessions.reader import SessionReader
from ash.sessions.types import (
    MessageEntry,
    SessionHeader,
    ToolResultEntry,
    ToolUseEntry,
    _sanitize,
    parse_entry,
    session_key,
)
from ash.sessions.writer import SessionWriter


class TestSessionKey:
    """Tests for session key generation."""

    def test_provider_only(self):
        key = session_key("cli")
        assert key == "cli"

    def test_provider_with_chat_id(self):
        key = session_key("telegram", chat_id="12345")
        assert key == "telegram_12345"

    def test_provider_with_user_id(self):
        key = session_key("api", user_id="user123")
        assert key == "api_user123"

    def test_chat_id_takes_precedence_over_user_id(self):
        key = session_key("api", chat_id="chat1", user_id="user1")
        assert key == "api_chat1"

    def test_sanitizes_special_characters(self):
        key = session_key("cli", chat_id="test@user.com")
        assert key == "cli_test_user_com"

    def test_limits_length(self):
        long_id = "a" * 100
        key = session_key("cli", chat_id=long_id)
        # Provider + _ + max 64 chars
        assert len(key) <= 68


class TestSanitize:
    """Tests for path sanitization."""

    def test_replaces_special_chars(self):
        assert _sanitize("hello@world.com") == "hello_world_com"

    def test_collapses_multiple_underscores(self):
        assert _sanitize("hello---world") == "hello_world"

    def test_strips_leading_trailing_underscores(self):
        assert _sanitize("__hello__") == "hello"

    def test_returns_default_for_empty(self):
        assert _sanitize("") == "default"
        assert _sanitize("___") == "default"

    def test_limits_length(self):
        long_str = "a" * 100
        result = _sanitize(long_str)
        assert len(result) <= 64


class TestSessionHeader:
    """Tests for SessionHeader entry."""

    def test_create(self):
        header = SessionHeader.create(
            provider="cli",
            user_id="user1",
            chat_id="chat1",
        )
        assert header.type == "session"
        assert header.provider == "cli"
        assert header.user_id == "user1"
        assert header.chat_id == "chat1"
        assert header.id  # Should have generated ID
        assert header.created_at  # Should have timestamp

    def test_to_dict(self):
        header = SessionHeader.create(provider="cli")
        data = header.to_dict()

        assert data["type"] == "session"
        assert data["version"] == "1"
        assert data["provider"] == "cli"
        assert "id" in data
        assert "created_at" in data

    def test_round_trip(self):
        original = SessionHeader.create(
            provider="telegram",
            user_id="user1",
            chat_id="chat1",
        )
        data = original.to_dict()
        restored = SessionHeader.from_dict(data)

        assert restored.id == original.id
        assert restored.provider == original.provider
        assert restored.user_id == original.user_id
        assert restored.chat_id == original.chat_id


class TestMessageEntry:
    """Tests for MessageEntry."""

    def test_create_user_message(self):
        entry = MessageEntry.create(
            role="user",
            content="Hello!",
            user_id="user1",
        )
        assert entry.type == "message"
        assert entry.role == "user"
        assert entry.content == "Hello!"
        assert entry.user_id == "user1"

    def test_create_assistant_message_with_blocks(self):
        content = [
            {"type": "text", "text": "Let me help you."},
            {
                "type": "tool_use",
                "id": "t1",
                "name": "bash",
                "input": {"command": "ls"},
            },
        ]
        entry = MessageEntry.create(role="assistant", content=content)

        assert entry.role == "assistant"
        assert len(entry.content) == 2

    def test_to_history_dict_extracts_text(self):
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {}},
        ]
        entry = MessageEntry.create(role="assistant", content=content)
        history = entry.to_history_dict()

        # Should only contain text, not tool_use
        assert history["content"] == "Hello"
        assert "type" not in history  # History format doesn't have type

    def test_round_trip(self):
        original = MessageEntry.create(
            role="user",
            content="Test message",
            token_count=10,
            user_id="user1",
        )
        data = original.to_dict()
        restored = MessageEntry.from_dict(data)

        assert restored.id == original.id
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.token_count == original.token_count


class TestToolUseEntry:
    """Tests for ToolUseEntry."""

    def test_create(self):
        entry = ToolUseEntry.create(
            tool_use_id="t1",
            message_id="m1",
            name="bash",
            input_data={"command": "ls"},
        )
        assert entry.type == "tool_use"
        assert entry.id == "t1"
        assert entry.message_id == "m1"
        assert entry.name == "bash"
        assert entry.input == {"command": "ls"}

    def test_round_trip(self):
        original = ToolUseEntry.create(
            tool_use_id="t1",
            message_id="m1",
            name="bash",
            input_data={"command": "ls -la"},
        )
        data = original.to_dict()
        restored = ToolUseEntry.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.input == original.input


class TestToolResultEntry:
    """Tests for ToolResultEntry."""

    def test_create_success(self):
        entry = ToolResultEntry.create(
            tool_use_id="t1",
            output="file1.txt\nfile2.txt",
            success=True,
            duration_ms=50,
        )
        assert entry.type == "tool_result"
        assert entry.tool_use_id == "t1"
        assert entry.output == "file1.txt\nfile2.txt"
        assert entry.success is True
        assert entry.duration_ms == 50

    def test_create_error(self):
        entry = ToolResultEntry.create(
            tool_use_id="t1",
            output="Command not found",
            success=False,
        )
        assert entry.success is False

    def test_round_trip(self):
        original = ToolResultEntry.create(
            tool_use_id="t1",
            output="output",
            success=True,
            duration_ms=100,
        )
        data = original.to_dict()
        restored = ToolResultEntry.from_dict(data)

        assert restored.tool_use_id == original.tool_use_id
        assert restored.output == original.output
        assert restored.success == original.success
        assert restored.duration_ms == original.duration_ms


class TestParseEntry:
    """Tests for parse_entry function."""

    def test_parse_session(self):
        data = {
            "type": "session",
            "version": "1",
            "id": "abc",
            "created_at": "2026-01-11T10:00:00+00:00",
            "provider": "cli",
        }
        entry = parse_entry(data)
        assert isinstance(entry, SessionHeader)
        assert entry.id == "abc"

    def test_parse_message(self):
        data = {
            "type": "message",
            "id": "m1",
            "role": "user",
            "content": "Hello",
            "created_at": "2026-01-11T10:00:00+00:00",
        }
        entry = parse_entry(data)
        assert isinstance(entry, MessageEntry)
        assert entry.content == "Hello"

    def test_parse_tool_use(self):
        data = {
            "type": "tool_use",
            "id": "t1",
            "message_id": "m1",
            "name": "bash",
            "input": {},
        }
        entry = parse_entry(data)
        assert isinstance(entry, ToolUseEntry)
        assert entry.name == "bash"

    def test_parse_tool_result(self):
        data = {
            "type": "tool_result",
            "tool_use_id": "t1",
            "output": "result",
            "success": True,
        }
        entry = parse_entry(data)
        assert isinstance(entry, ToolResultEntry)
        assert entry.output == "result"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown entry type"):
            parse_entry({"type": "unknown"})


class TestSessionWriter:
    """Tests for SessionWriter."""

    @pytest.fixture
    def session_dir(self, tmp_path):
        return tmp_path / "test_session"

    @pytest.fixture
    def writer(self, session_dir):
        return SessionWriter(session_dir)

    @pytest.mark.asyncio
    async def test_creates_directory(self, writer, session_dir):
        assert not session_dir.exists()
        await writer.ensure_directory()
        assert session_dir.exists()

    @pytest.mark.asyncio
    async def test_write_header(self, writer, session_dir):
        header = SessionHeader.create(provider="cli")
        await writer.write_header(header)

        # Verify context.jsonl was created
        context_file = session_dir / "context.jsonl"
        assert context_file.exists()

        # Verify content
        data = json.loads(context_file.read_text().strip().split("\n")[0])
        assert data["type"] == "session"
        assert data["provider"] == "cli"

    @pytest.mark.asyncio
    async def test_write_message_to_both_files(self, writer, session_dir):
        entry = MessageEntry.create(role="user", content="Hello!")
        await writer.write_message(entry)

        # Check context.jsonl
        context_file = session_dir / "context.jsonl"
        assert context_file.exists()
        data = json.loads(context_file.read_text().strip().split("\n")[0])
        assert data["type"] == "message"
        assert data["content"] == "Hello!"

        # Check history.jsonl
        history_file = session_dir / "history.jsonl"
        assert history_file.exists()
        data = json.loads(history_file.read_text().strip().split("\n")[0])
        assert "type" not in data  # History format
        assert data["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_write_tool_use_context_only(self, writer, session_dir):
        entry = ToolUseEntry.create(
            tool_use_id="t1",
            message_id="m1",
            name="bash",
            input_data={"command": "ls"},
        )
        await writer.write_tool_use(entry)

        # Check context.jsonl
        context_file = session_dir / "context.jsonl"
        assert context_file.exists()

        # History should not exist (or be empty)
        history_file = session_dir / "history.jsonl"
        assert not history_file.exists()

    @pytest.mark.asyncio
    async def test_write_tool_result_context_only(self, writer, session_dir):
        entry = ToolResultEntry.create(
            tool_use_id="t1",
            output="output",
            success=True,
        )
        await writer.write_tool_result(entry)

        context_file = session_dir / "context.jsonl"
        assert context_file.exists()

        history_file = session_dir / "history.jsonl"
        assert not history_file.exists()


class TestSessionReader:
    """Tests for SessionReader."""

    @pytest.fixture
    def session_dir(self, tmp_path):
        return tmp_path / "test_session"

    @pytest.fixture
    def reader(self, session_dir):
        return SessionReader(session_dir)

    def test_exists_false_when_no_file(self, reader):
        assert not reader.exists()

    def test_exists_true_when_file_exists(self, reader, session_dir):
        session_dir.mkdir(parents=True)
        (session_dir / "context.jsonl").write_text('{"type":"session"}\n')
        assert reader.exists()

    @pytest.mark.asyncio
    async def test_load_entries_empty_when_no_file(self, reader):
        entries = await reader.load_entries()
        assert entries == []

    @pytest.mark.asyncio
    async def test_load_entries_parses_all_types(self, reader, session_dir):
        session_dir.mkdir(parents=True)
        context_file = session_dir / "context.jsonl"

        lines = [
            '{"type":"session","version":"1","id":"s1","created_at":"2026-01-11T10:00:00+00:00","provider":"cli"}',
            '{"type":"message","id":"m1","role":"user","content":"Hello","created_at":"2026-01-11T10:00:01+00:00"}',
            '{"type":"tool_use","id":"t1","message_id":"m1","name":"bash","input":{}}',
            '{"type":"tool_result","tool_use_id":"t1","output":"result","success":true}',
        ]
        context_file.write_text("\n".join(lines) + "\n")

        entries = await reader.load_entries()
        assert len(entries) == 4
        assert isinstance(entries[0], SessionHeader)
        assert isinstance(entries[1], MessageEntry)
        assert isinstance(entries[2], ToolUseEntry)
        assert isinstance(entries[3], ToolResultEntry)

    @pytest.mark.asyncio
    async def test_load_header(self, reader, session_dir):
        session_dir.mkdir(parents=True)
        context_file = session_dir / "context.jsonl"
        context_file.write_text(
            '{"type":"session","version":"1","id":"s1","created_at":"2026-01-11T10:00:00+00:00","provider":"cli"}\n'
        )

        header = await reader.load_header()
        assert header is not None
        assert header.id == "s1"
        assert header.provider == "cli"

    @pytest.mark.asyncio
    async def test_load_messages_for_llm_converts_to_message_objects(
        self, reader, session_dir
    ):
        session_dir.mkdir(parents=True)
        context_file = session_dir / "context.jsonl"

        lines = [
            '{"type":"session","version":"1","id":"s1","created_at":"2026-01-11T10:00:00+00:00","provider":"cli"}',
            '{"type":"message","id":"m1","role":"user","content":"Hello","created_at":"2026-01-11T10:00:01+00:00","token_count":5}',
            '{"type":"message","id":"m2","role":"assistant","content":"Hi!","created_at":"2026-01-11T10:00:02+00:00","token_count":5}',
        ]
        context_file.write_text("\n".join(lines) + "\n")

        messages, ids = await reader.load_messages_for_llm()

        assert len(messages) == 2
        assert messages[0].role.value == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role.value == "assistant"
        assert ids == ["m1", "m2"]


class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.fixture
    def sessions_path(self, tmp_path):
        return tmp_path / "sessions"

    @pytest.fixture
    def manager(self, sessions_path):
        return SessionManager(
            provider="cli",
            sessions_path=sessions_path,
        )

    def test_session_key_computed(self, manager):
        assert manager.session_key == "cli"

    def test_session_dir_path(self, manager, sessions_path):
        assert manager.session_dir == sessions_path / "cli"

    def test_exists_false_initially(self, manager):
        assert not manager.exists()

    @pytest.mark.asyncio
    async def test_ensure_session_creates_header(self, manager, sessions_path):
        header = await manager.ensure_session()

        assert header.provider == "cli"
        assert header.id

        # File should exist
        context_file = sessions_path / "cli" / "context.jsonl"
        assert context_file.exists()

    @pytest.mark.asyncio
    async def test_add_user_message(self, manager, sessions_path):
        await manager.ensure_session()
        msg_id = await manager.add_user_message("Hello!")

        assert msg_id

        # Verify written to both files
        context_file = sessions_path / "cli" / "context.jsonl"
        history_file = sessions_path / "cli" / "history.jsonl"

        assert context_file.exists()
        assert history_file.exists()

    @pytest.mark.asyncio
    async def test_add_assistant_message_with_tool_use(self, manager, sessions_path):
        await manager.ensure_session()

        content = [
            TextContent(text="Let me check."),
            ToolUse(id="t1", name="bash", input={"command": "ls"}),
        ]
        msg_id = await manager.add_assistant_message(content)

        assert msg_id

        # Verify context has both message and tool_use entries
        context_file = sessions_path / "cli" / "context.jsonl"
        lines = [
            json.loads(line)
            for line in context_file.read_text().strip().split("\n")
            if line.strip()
        ]

        # Should have: session, message, tool_use
        types = [line["type"] for line in lines]
        assert "session" in types
        assert "message" in types
        assert "tool_use" in types

    @pytest.mark.asyncio
    async def test_add_tool_result(self, manager, sessions_path):
        await manager.ensure_session()
        await manager.add_tool_result(
            tool_use_id="t1",
            output="file1.txt",
            success=True,
            duration_ms=50,
        )

        context_file = sessions_path / "cli" / "context.jsonl"
        lines = [
            json.loads(line)
            for line in context_file.read_text().strip().split("\n")
            if line.strip()
        ]

        tool_results = [entry for entry in lines if entry["type"] == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0]["output"] == "file1.txt"
        assert tool_results[0]["duration_ms"] == 50

    @pytest.mark.asyncio
    async def test_load_messages_for_llm(self, manager):
        await manager.ensure_session()
        await manager.add_user_message("Hello")
        await manager.add_assistant_message("Hi there!")
        await manager.add_user_message("How are you?")

        messages, ids = await manager.load_messages_for_llm()

        assert len(messages) == 3
        assert len(ids) == 3
        assert messages[0].content == "Hello"
        assert messages[2].content == "How are you?"

    @pytest.mark.asyncio
    async def test_full_conversation_lifecycle(self, manager):
        """Test a complete conversation with tool use."""
        await manager.ensure_session()

        # User asks something
        await manager.add_user_message("List files in current directory")

        # Assistant responds with tool use
        content = [
            TextContent(text="Let me check that for you."),
            ToolUse(id="t1", name="bash", input={"command": "ls"}),
        ]
        await manager.add_assistant_message(content)

        # Tool result
        await manager.add_tool_result(
            tool_use_id="t1",
            output="file1.txt\nfile2.txt",
            success=True,
        )

        # Assistant final response
        await manager.add_assistant_message("I found 2 files: file1.txt and file2.txt")

        # Verify we can load it all back
        messages, _ = await manager.load_messages_for_llm()

        # Should have: user, assistant (with tool), tool_result (as user), assistant
        assert len(messages) >= 3  # At least user, assistant, assistant

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, sessions_path):
        sessions = await SessionManager.list_sessions(sessions_path)
        assert sessions == []

    @pytest.mark.asyncio
    async def test_list_sessions_returns_all(self, sessions_path):
        # Create two sessions
        m1 = SessionManager(provider="cli", sessions_path=sessions_path)
        await m1.ensure_session()

        m2 = SessionManager(
            provider="telegram", chat_id="123", sessions_path=sessions_path
        )
        await m2.ensure_session()

        sessions = await SessionManager.list_sessions(sessions_path)

        assert len(sessions) == 2
        providers = {s["provider"] for s in sessions}
        assert providers == {"cli", "telegram"}

    @pytest.mark.asyncio
    async def test_get_session_by_key(self, sessions_path):
        # Create a session
        m1 = SessionManager(
            provider="telegram", chat_id="123", sessions_path=sessions_path
        )
        await m1.ensure_session()
        await m1.add_user_message("Test")

        # Retrieve by key
        m2 = await SessionManager.get_session("telegram_123", sessions_path)

        assert m2 is not None
        assert m2.provider == "telegram"
        assert m2.chat_id == "123"

        # Verify can load messages
        messages, _ = await m2.load_messages_for_llm()
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, sessions_path):
        result = await SessionManager.get_session("nonexistent", sessions_path)
        assert result is None
