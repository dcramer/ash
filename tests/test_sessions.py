"""Tests for JSONL session management."""

from __future__ import annotations

import json

import pytest

from ash.llm.types import TextContent, ToolUse
from ash.sessions.manager import SessionManager
from ash.sessions.reader import SessionReader
from ash.sessions.types import (
    MessageEntry,
    ToolResultEntry,
    ToolUseEntry,
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

    def test_long_ids_include_hash_suffix(self):
        """Long IDs are hashed to avoid silent key collisions."""
        id1 = "a" * 100
        id2 = "a" * 99 + "b"
        key1 = session_key("cli", chat_id=id1)
        key2 = session_key("cli", chat_id=id2)
        assert key1 != key2
        assert key1.startswith("cli_")
        assert key2.startswith("cli_")


class TestParseEntry:
    """Tests for entry parsing error handling."""

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown entry type"):
            parse_entry({"type": "unknown"})

    def test_missing_type_raises(self):
        with pytest.raises(KeyError, match="type"):
            parse_entry({})

    def test_session_header_requires_version(self):
        with pytest.raises(KeyError, match="version"):
            parse_entry(
                {
                    "type": "session",
                    "id": "s1",
                    "created_at": "2026-01-11T10:00:00+00:00",
                    "provider": "cli",
                }
            )

    def test_session_header_rejects_null_created_at(self):
        with pytest.raises(TypeError):
            parse_entry(
                {
                    "type": "session",
                    "version": "2",
                    "id": "s1",
                    "created_at": None,
                    "provider": "cli",
                }
            )

    def test_compaction_entry_requires_created_at(self):
        with pytest.raises(KeyError, match="created_at"):
            parse_entry(
                {
                    "type": "compaction",
                    "id": "c1",
                    "summary": "summary",
                    "tokens_before": 100,
                    "tokens_after": 50,
                    "first_kept_entry_id": "m1",
                }
            )

    def test_agent_session_complete_requires_created_at(self):
        with pytest.raises(KeyError, match="created_at"):
            parse_entry(
                {
                    "type": "agent_session_complete",
                    "agent_session_id": "a1",
                    "result": "done",
                    "is_error": False,
                }
            )


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

    @pytest.mark.asyncio
    async def test_get_messages_around_requires_internal_message_id(
        self, reader, session_dir
    ):
        """get_messages_around only matches stored message IDs."""
        session_dir.mkdir(parents=True)
        lines = [
            '{"type":"session","version":"1","id":"s1","created_at":"2026-01-11T10:00:00+00:00","provider":"cli"}',
            '{"type":"message","id":"m1","role":"user","content":"Hello","created_at":"2026-01-11T10:00:01+00:00","metadata":{"external_id":"ext-1"}}',
            '{"type":"message","id":"m2","role":"assistant","content":"Hi!","created_at":"2026-01-11T10:00:02+00:00","metadata":{"external_id":"ext-2"}}',
        ]
        (session_dir / "context.jsonl").write_text("\n".join(lines) + "\n")

        by_internal = await reader.get_messages_around("m1")
        assert len(by_internal) == 2

        by_external = await reader.get_messages_around("ext-1")
        assert by_external == []

    @pytest.mark.asyncio
    async def test_external_id_lookup_uses_external_id_only(self, reader, session_dir):
        session_dir.mkdir(parents=True)
        lines = [
            '{"type":"session","version":"1","id":"s1","created_at":"2026-01-11T10:00:00+00:00","provider":"cli"}',
            '{"type":"message","id":"m1","role":"user","content":"Hello","created_at":"2026-01-11T10:00:01+00:00","metadata":{"external_id":"ext-user"}}',
            '{"type":"message","id":"m2","role":"assistant","content":"Hi!","created_at":"2026-01-11T10:00:02+00:00","metadata":{"bot_response_id":"ext-legacy"}}',
            '{"type":"message","id":"m3","role":"assistant","content":"Hello again","created_at":"2026-01-11T10:00:03+00:00","metadata":{"external_id":"ext-bot"}}',
        ]
        (session_dir / "context.jsonl").write_text("\n".join(lines) + "\n")

        assert await reader.has_message_with_external_id("ext-user")
        assert await reader.has_message_with_external_id("ext-bot")
        assert not await reader.has_message_with_external_id("ext-legacy")

        assert (await reader.get_message_by_external_id("ext-user")) is not None
        assert (await reader.get_message_by_external_id("ext-bot")) is not None
        assert (await reader.get_message_by_external_id("ext-legacy")) is None


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
    async def test_list_sessions(self, sessions_path):
        """Can list all sessions."""
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

    @pytest.mark.asyncio
    async def test_get_recent_message_ids_uses_chronological_order(self, manager):
        """Recent message IDs are selected by append order, not set iteration order."""
        await manager.ensure_session()

        await manager.add_user_message("one")
        m2 = await manager.add_assistant_message("two")
        m3 = await manager.add_user_message("three")

        recent = await manager.get_recent_message_ids(recency_window=2)
        assert recent == {m2, m3}


class TestSessionBranching:
    """Tests for tree-structured conversation branching."""

    @pytest.fixture
    def sessions_path(self, tmp_path):
        return tmp_path / "sessions"

    @pytest.fixture
    def manager(self, sessions_path):
        return SessionManager(provider="cli", sessions_path=sessions_path)

    @pytest.mark.asyncio
    async def test_parent_id_auto_set(self, manager):
        """Messages automatically chain via parent_id."""
        await manager.ensure_session()

        m1_id = await manager.add_user_message("Hello")
        m2_id = await manager.add_assistant_message("Hi there!")
        m3_id = await manager.add_user_message("How are you?")

        entries = await manager._reader.load_entries()
        msgs = [e for e in entries if isinstance(e, MessageEntry)]

        assert msgs[0].id == m1_id
        assert msgs[0].parent_id is None  # First message has no parent

        assert msgs[1].id == m2_id
        assert msgs[1].parent_id == m1_id

        assert msgs[2].id == m3_id
        assert msgs[2].parent_id == m2_id

    @pytest.mark.asyncio
    async def test_fork_at_message(self, manager):
        """Forking creates a new branch and sets current position."""
        await manager.ensure_session()

        await manager.add_user_message("Message 1")
        m2_id = await manager.add_assistant_message("Response 1")
        await manager.add_user_message("Message 2")
        await manager.add_assistant_message("Response 2")

        # Fork at m2 (after first exchange)
        branch_id = manager.fork_at_message(m2_id)
        assert branch_id is not None

        # Current message should be set to fork point
        assert manager._current_message_id == m2_id

        # State should have branches
        state = manager._load_state()
        assert state is not None
        assert len(state.branches) == 2  # main + new branch

        # New message should chain from fork point
        m5_id = await manager.add_user_message("Branched message")
        entries = await manager._reader.load_entries()
        msgs = {e.id: e for e in entries if isinstance(e, MessageEntry)}
        assert msgs[m5_id].parent_id == m2_id

    @pytest.mark.asyncio
    async def test_branch_aware_loading(self, manager):
        """Branch-aware load returns only messages on the branch path."""
        await manager.ensure_session()

        # Build linear conversation: M1 -> M2 -> M3 -> M4
        await manager.add_user_message("M1")
        m2_id = await manager.add_assistant_message("M2")
        await manager.add_user_message("M3")
        await manager.add_assistant_message("M4")

        # Fork at M2, add M5
        branch_id = manager.fork_at_message(m2_id)
        m5_id = await manager.add_user_message("M5 (branched)")

        # Load branch: should see M1, M2, M5 — not M3, M4
        messages, msg_ids = await manager.load_messages_for_llm(
            branch_head_id=m5_id, branch_id=branch_id
        )

        contents = [
            m.content if isinstance(m.content, str) else str(m.content)
            for m in messages
        ]
        assert "M1" in contents[0]
        assert "M2" in contents[1]
        assert "M5 (branched)" in contents[2]
        assert len(messages) == 3

    @pytest.mark.asyncio
    async def test_nested_fork(self, manager):
        """Multiple forks from the same point create independent branches."""
        await manager.ensure_session()

        await manager.add_user_message("M1")
        m2_id = await manager.add_assistant_message("M2")
        await manager.add_user_message("M3")
        await manager.add_assistant_message("M4")

        # Fork 1 at M2
        branch1_id = manager.fork_at_message(m2_id)
        await manager.add_user_message("Branch1-M5")
        m6_id = await manager.add_assistant_message("Branch1-M6")

        # Fork 2 at M2 (again, different branch)
        branch2_id = manager.fork_at_message(m2_id)
        m7_id = await manager.add_user_message("Branch2-M7")

        # Load branch 1: M1, M2, M5, M6
        msgs1, _ = await manager.load_messages_for_llm(
            branch_head_id=m6_id, branch_id=branch1_id
        )
        contents1 = [
            m.content if isinstance(m.content, str) else str(m.content) for m in msgs1
        ]
        assert len(msgs1) == 4
        assert "M1" in contents1[0]
        assert "M2" in contents1[1]
        assert "Branch1-M5" in contents1[2]
        assert "Branch1-M6" in contents1[3]

        # Load branch 2: M1, M2, M7
        msgs2, _ = await manager.load_messages_for_llm(
            branch_head_id=m7_id, branch_id=branch2_id
        )
        contents2 = [
            m.content if isinstance(m.content, str) else str(m.content) for m in msgs2
        ]
        assert len(msgs2) == 3
        assert "M1" in contents2[0]
        assert "M2" in contents2[1]
        assert "Branch2-M7" in contents2[2]

    @pytest.mark.asyncio
    async def test_branch_with_tool_pairs(self, manager):
        """Tool use/result pairs stay with their branch."""
        await manager.ensure_session()

        await manager.add_user_message("M1")
        await manager.add_assistant_message(
            [
                TextContent(text="Let me check."),
                ToolUse(id="t1", name="bash", input={"command": "ls"}),
            ]
        )
        await manager.add_tool_result(tool_use_id="t1", output="files", success=True)
        m3_id = await manager.add_assistant_message("Found files")
        await manager.add_user_message("M4")
        await manager.add_assistant_message(
            [
                TextContent(text="Checking more."),
                ToolUse(id="t2", name="bash", input={"command": "pwd"}),
            ]
        )
        await manager.add_tool_result(tool_use_id="t2", output="/home", success=True)
        await manager.add_assistant_message("Done")

        # Fork at m3 (after first tool exchange)
        branch_id = manager.fork_at_message(m3_id)
        m7_id = await manager.add_user_message("Branched question")

        # Load branch: should see M1, M2(with tool), tool_result, M3, M7
        # but NOT M4, M5(with t2), t2 result, M6
        messages, _ = await manager.load_messages_for_llm(
            branch_head_id=m7_id, branch_id=branch_id
        )

        # Check that t1 tool result is present (on branch) but t2 is not
        has_t1_result = False
        has_t2_result = False
        for msg in messages:
            if isinstance(msg.content, list):
                for block in msg.content:
                    from ash.llm.types import ToolResult

                    if isinstance(block, ToolResult):
                        if block.tool_use_id == "t1":
                            has_t1_result = True
                        if block.tool_use_id == "t2":
                            has_t2_result = True

        assert has_t1_result, "t1 tool result should be on branch"
        assert not has_t2_result, "t2 tool result should NOT be on branch"

    @pytest.mark.asyncio
    async def test_branch_head_tracking(self, manager):
        """Branch head updates after writes."""
        await manager.ensure_session()

        await manager.add_user_message("M1")
        m2_id = await manager.add_assistant_message("M2")
        await manager.add_user_message("M3")

        branch_id = manager.fork_at_message(m2_id)
        m4_id = await manager.add_user_message("Branched")

        # Update branch head
        manager.update_branch_head(branch_id, m4_id)

        branch = manager.get_branch_for_message(m4_id)
        assert branch is not None
        assert branch.branch_id == branch_id
        assert branch.head_message_id == m4_id
        assert branch.fork_point_id == m2_id

    @pytest.mark.asyncio
    async def test_linear_load_unchanged(self, manager):
        """Linear loading (no branch) still works as before."""
        await manager.ensure_session()

        await manager.add_user_message("Hello")
        await manager.add_assistant_message("Hi!")
        await manager.add_user_message("More")

        messages, ids = await manager.load_messages_for_llm()
        assert len(messages) == 3

    @pytest.mark.asyncio
    async def test_parent_id_serialization(self, manager):
        """parent_id survives serialization roundtrip."""
        await manager.ensure_session()

        m1_id = await manager.add_user_message("M1")
        await manager.add_assistant_message("M2")

        # Read raw JSONL
        context_file = manager._session_dir / "context.jsonl"
        lines = context_file.read_text().strip().split("\n")

        # m2 should have parent_id serialized
        m2_data = json.loads(lines[2])  # header, m1, m2
        assert m2_data["parent_id"] == m1_id

        # m1 should not have parent_id (None = omitted)
        m1_data = json.loads(lines[1])
        assert "parent_id" not in m1_data


class TestSubagentIsolation:
    """Tests for subagent session isolation — entries route to subagents/ dir."""

    @pytest.fixture
    def sessions_path(self, tmp_path):
        return tmp_path / "sessions"

    @pytest.fixture
    def manager(self, sessions_path):
        return SessionManager(provider="cli", sessions_path=sessions_path)

    @pytest.mark.asyncio
    async def test_subagent_messages_go_to_subagent_file(self, manager):
        """Messages with agent_session_id write to subagents/{id}.jsonl, not context.jsonl."""
        await manager.ensure_session()

        # Main agent message — goes to context.jsonl
        await manager.add_user_message("Main message")

        # Subagent message — goes to subagents/sub1.jsonl
        await manager.add_user_message("Subagent msg", agent_session_id="sub1")

        # context.jsonl has only the main message (+ header)
        context_lines = (
            (manager._session_dir / "context.jsonl").read_text().strip().split("\n")
        )
        context_entries = [json.loads(line) for line in context_lines]
        messages = [e for e in context_entries if e["type"] == "message"]
        assert len(messages) == 1
        assert messages[0]["content"] == "Main message"

        # subagents/sub1.jsonl has the subagent message
        subagent_file = manager._session_dir / "subagents" / "sub1.jsonl"
        assert subagent_file.exists()
        sub_lines = subagent_file.read_text().strip().split("\n")
        sub_entries = [json.loads(line) for line in sub_lines]
        assert len(sub_entries) == 1
        assert sub_entries[0]["content"] == "Subagent msg"

    @pytest.mark.asyncio
    async def test_subagent_tool_use_goes_to_subagent_file(self, manager):
        """Tool use with agent_session_id routes to subagent file."""
        await manager.ensure_session()

        await manager.add_tool_use(
            tool_use_id="t1",
            name="bash",
            input_data={"command": "ls"},
            agent_session_id="sub1",
        )

        # Not in context.jsonl
        context = (manager._session_dir / "context.jsonl").read_text()
        assert "tool_use" not in context or '"type":"session"' in context

        # In subagent file
        subagent_file = manager._session_dir / "subagents" / "sub1.jsonl"
        assert subagent_file.exists()
        entry = json.loads(subagent_file.read_text().strip())
        assert entry["type"] == "tool_use"
        assert entry["name"] == "bash"

    @pytest.mark.asyncio
    async def test_subagent_tool_result_goes_to_subagent_file(self, manager):
        """Tool result with agent_session_id routes to subagent file."""
        await manager.ensure_session()

        await manager.add_tool_result(
            tool_use_id="t1",
            output="file1.txt",
            success=True,
            agent_session_id="sub1",
        )

        # In subagent file
        subagent_file = manager._session_dir / "subagents" / "sub1.jsonl"
        assert subagent_file.exists()
        entry = json.loads(subagent_file.read_text().strip())
        assert entry["type"] == "tool_result"
        assert entry["output"] == "file1.txt"

    @pytest.mark.asyncio
    async def test_subagent_assistant_with_tool_use_content(self, manager):
        """Assistant message with ContentBlock tool uses routes to subagent file."""
        await manager.ensure_session()

        await manager.add_assistant_message(
            [
                TextContent(text="Checking..."),
                ToolUse(id="t1", name="bash", input={"command": "ls"}),
            ],
            agent_session_id="sub1",
        )

        # subagent file should have both the message and the auto-extracted tool_use
        subagent_file = manager._session_dir / "subagents" / "sub1.jsonl"
        lines = subagent_file.read_text().strip().split("\n")
        entries = [json.loads(line) for line in lines]

        types = [e["type"] for e in entries]
        assert "message" in types
        assert "tool_use" in types

    @pytest.mark.asyncio
    async def test_agent_session_marker_stays_in_context(self, manager):
        """AgentSessionEntry marker always goes to context.jsonl (parent timeline)."""
        await manager.ensure_session()

        agent_session_id = await manager.start_agent_session(
            parent_tool_use_id="tu1",
            agent_type="skill",
            agent_name="debug-self",
        )

        # Marker in context.jsonl
        context_lines = (
            (manager._session_dir / "context.jsonl").read_text().strip().split("\n")
        )
        context_entries = [json.loads(line) for line in context_lines]
        agent_entries = [e for e in context_entries if e["type"] == "agent_session"]
        assert len(agent_entries) == 1
        assert agent_entries[0]["id"] == agent_session_id


class TestDMThreading:
    """Tests for DM threading via resolve_reply_chain_thread."""

    @pytest.fixture
    def handler(self, tmp_path):
        from ash.config.models import ConversationConfig
        from ash.providers.telegram.handlers.session_handler import SessionHandler

        return SessionHandler(
            provider_name="telegram",
            config=None,
            conversation_config=ConversationConfig(),
        )

    def _make_message(
        self,
        msg_id: str,
        chat_id: str = "user123",
        chat_type: str = "private",
        reply_to: str | None = None,
    ):
        from ash.providers.base import IncomingMessage

        return IncomingMessage(
            id=msg_id,
            chat_id=chat_id,
            user_id="user123",
            text="test",
            reply_to_message_id=reply_to,
            metadata={"chat_type": chat_type},
        )

    @pytest.mark.asyncio
    async def test_standalone_dm_creates_new_thread(self, handler):
        """A standalone DM (no reply) gets a new thread_id."""
        msg = self._make_message("100", chat_type="private")
        thread_id = await handler.resolve_reply_chain_thread(msg)
        assert thread_id is not None

    @pytest.mark.asyncio
    async def test_dm_reply_joins_parent_thread(self, handler):
        """A DM reply to a known message joins the parent's thread."""
        # First message establishes a thread
        msg1 = self._make_message("100", chat_type="private")
        thread1 = await handler.resolve_reply_chain_thread(msg1)

        # Reply to first message should join same thread
        msg2 = self._make_message("101", chat_type="private", reply_to="100")
        thread2 = await handler.resolve_reply_chain_thread(msg2)

        assert thread2 == thread1

    @pytest.mark.asyncio
    async def test_group_message_still_gets_thread(self, handler):
        """Group messages continue to get thread IDs (existing behavior)."""
        msg = self._make_message("100", chat_type="group")
        thread_id = await handler.resolve_reply_chain_thread(msg)
        assert thread_id is not None

    @pytest.mark.asyncio
    async def test_dm_reply_to_unknown_parent_creates_thread(self, handler):
        """Reply to an unknown message still gets a concrete thread ID."""
        msg = self._make_message("101", chat_type="private", reply_to="50")
        thread_id = await handler.resolve_reply_chain_thread(msg)
        assert thread_id is not None


class TestChatHistoryInjection:
    """Tests for removal of cross-thread chat history injection."""

    def test_session_context_has_no_chat_history_field(self):
        """SessionContext should not have a chat_history field (removed)."""
        from ash.core.session import SessionContext

        ctx = SessionContext()
        assert not hasattr(ctx, "chat_history")
