"""Tests for memory store operations."""

from datetime import UTC, datetime, timedelta

import pytest


class TestSessionOperations:
    """Tests for session management."""

    async def test_get_or_create_session_creates_new(self, memory_store):
        session = await memory_store.get_or_create_session(
            provider="telegram",
            chat_id="chat-123",
            user_id="user-456",
        )
        assert session.id is not None
        assert session.provider == "telegram"
        assert session.chat_id == "chat-123"
        assert session.user_id == "user-456"

    async def test_get_or_create_session_returns_existing(self, memory_store):
        # Create first session
        session1 = await memory_store.get_or_create_session(
            provider="telegram",
            chat_id="chat-123",
            user_id="user-456",
        )
        # Get same session again
        session2 = await memory_store.get_or_create_session(
            provider="telegram",
            chat_id="chat-123",
            user_id="user-456",
        )
        assert session1.id == session2.id

    async def test_get_or_create_session_with_metadata(self, memory_store):
        session = await memory_store.get_or_create_session(
            provider="telegram",
            chat_id="chat-123",
            user_id="user-456",
            metadata={"custom": "data"},
        )
        assert session.metadata_ == {"custom": "data"}

    async def test_get_session_by_id(self, memory_store):
        created = await memory_store.get_or_create_session(
            provider="test",
            chat_id="chat-1",
            user_id="user-1",
        )
        retrieved = await memory_store.get_session(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id

    async def test_get_session_not_found(self, memory_store):
        result = await memory_store.get_session("nonexistent-id")
        assert result is None


class TestMessageOperations:
    """Tests for message storage and retrieval."""

    @pytest.fixture
    async def session_with_messages(self, memory_store):
        session = await memory_store.get_or_create_session(
            provider="test",
            chat_id="chat-1",
            user_id="user-1",
        )
        # Add messages with explicit timestamps for ordering
        await memory_store.add_message(
            session_id=session.id,
            role="user",
            content="Hello",
        )
        await memory_store.add_message(
            session_id=session.id,
            role="assistant",
            content="Hi there!",
        )
        await memory_store.add_message(
            session_id=session.id,
            role="user",
            content="How are you?",
        )
        return session

    async def test_add_message(self, memory_store):
        session = await memory_store.get_or_create_session(
            provider="test", chat_id="chat-1", user_id="user-1"
        )
        message = await memory_store.add_message(
            session_id=session.id,
            role="user",
            content="Hello, world!",
        )
        assert message.id is not None
        assert message.role == "user"
        assert message.content == "Hello, world!"

    async def test_add_message_with_metadata(self, memory_store):
        session = await memory_store.get_or_create_session(
            provider="test", chat_id="chat-1", user_id="user-1"
        )
        message = await memory_store.add_message(
            session_id=session.id,
            role="assistant",
            content="Response",
            token_count=50,
            metadata={"model": "test-model"},
        )
        assert message.token_count == 50
        assert message.metadata_ == {"model": "test-model"}

    async def test_get_messages(self, session_with_messages, memory_store):
        messages = await memory_store.get_messages(session_with_messages.id)
        assert len(messages) == 3
        # Should be oldest first
        assert messages[0].content == "Hello"
        assert messages[2].content == "How are you?"

    async def test_get_messages_with_limit(self, session_with_messages, memory_store):
        messages = await memory_store.get_messages(session_with_messages.id, limit=2)
        assert len(messages) == 2

    async def test_get_messages_empty_session(self, memory_store):
        session = await memory_store.get_or_create_session(
            provider="test", chat_id="chat-empty", user_id="user-1"
        )
        messages = await memory_store.get_messages(session.id)
        assert messages == []


class TestKnowledgeOperations:
    """Tests for knowledge base operations."""

    async def test_add_knowledge(self, memory_store):
        knowledge = await memory_store.add_knowledge(
            content="Python is a programming language.",
            source="manual",
        )
        assert knowledge.id is not None
        assert knowledge.content == "Python is a programming language."
        assert knowledge.source == "manual"

    async def test_add_knowledge_with_expiry(self, memory_store):
        expires = datetime.now(UTC) + timedelta(days=7)
        knowledge = await memory_store.add_knowledge(
            content="Temporary knowledge",
            expires_at=expires,
        )
        assert knowledge.expires_at == expires

    async def test_get_knowledge(self, memory_store):
        await memory_store.add_knowledge(content="Fact 1")
        await memory_store.add_knowledge(content="Fact 2")

        knowledge = await memory_store.get_knowledge()
        assert len(knowledge) == 2

    async def test_get_knowledge_excludes_expired(self, memory_store):
        # Add expired knowledge
        past = datetime.now(UTC) - timedelta(days=1)
        await memory_store.add_knowledge(
            content="Expired fact",
            expires_at=past,
        )
        # Add valid knowledge
        await memory_store.add_knowledge(content="Valid fact")

        knowledge = await memory_store.get_knowledge(include_expired=False)
        assert len(knowledge) == 1
        assert knowledge[0].content == "Valid fact"

    async def test_get_knowledge_includes_expired(self, memory_store):
        past = datetime.now(UTC) - timedelta(days=1)
        await memory_store.add_knowledge(content="Expired", expires_at=past)
        await memory_store.add_knowledge(content="Valid")

        knowledge = await memory_store.get_knowledge(include_expired=True)
        assert len(knowledge) == 2


class TestUserProfileOperations:
    """Tests for user profile management."""

    async def test_get_or_create_user_profile_creates_new(self, memory_store):
        profile = await memory_store.get_or_create_user_profile(
            user_id="user-123",
            provider="telegram",
            username="testuser",
            display_name="Test User",
        )
        assert profile.user_id == "user-123"
        assert profile.provider == "telegram"
        assert profile.username == "testuser"
        assert profile.display_name == "Test User"

    async def test_get_or_create_user_profile_updates_existing(self, memory_store):
        # Create profile
        await memory_store.get_or_create_user_profile(
            user_id="user-123",
            provider="telegram",
            username="oldname",
        )
        # Update with new username
        profile = await memory_store.get_or_create_user_profile(
            user_id="user-123",
            provider="telegram",
            username="newname",
        )
        assert profile.username == "newname"

    async def test_update_user_notes(self, memory_store):
        await memory_store.get_or_create_user_profile(
            user_id="user-123",
            provider="telegram",
        )
        profile = await memory_store.update_user_notes(
            user_id="user-123",
            notes="Prefers formal language",
        )
        assert profile is not None
        assert profile.notes == "Prefers formal language"

    async def test_update_user_notes_nonexistent(self, memory_store):
        result = await memory_store.update_user_notes(
            user_id="nonexistent",
            notes="Some notes",
        )
        assert result is None


class TestToolExecutionOperations:
    """Tests for tool execution logging."""

    async def test_log_tool_execution(self, memory_store):
        execution = await memory_store.log_tool_execution(
            tool_name="bash",
            input_data={"command": "ls -la"},
            output="file1.txt\nfile2.txt",
            success=True,
            duration_ms=150,
        )
        assert execution.id is not None
        assert execution.tool_name == "bash"
        assert execution.success is True
        assert execution.duration_ms == 150

    async def test_log_tool_execution_with_session(self, memory_store):
        session = await memory_store.get_or_create_session(
            provider="test", chat_id="chat-1", user_id="user-1"
        )
        execution = await memory_store.log_tool_execution(
            tool_name="bash",
            input_data={"command": "echo hello"},
            output="hello",
            success=True,
            session_id=session.id,
        )
        assert execution.session_id == session.id

    async def test_log_failed_execution(self, memory_store):
        execution = await memory_store.log_tool_execution(
            tool_name="bash",
            input_data={"command": "invalid"},
            output="Command not found",
            success=False,
        )
        assert execution.success is False

    async def test_get_tool_executions(self, memory_store):
        await memory_store.log_tool_execution(
            tool_name="bash", input_data={}, output="", success=True
        )
        await memory_store.log_tool_execution(
            tool_name="web_search", input_data={}, output="", success=True
        )

        executions = await memory_store.get_tool_executions()
        assert len(executions) == 2

    async def test_get_tool_executions_by_name(self, memory_store):
        await memory_store.log_tool_execution(
            tool_name="bash", input_data={}, output="", success=True
        )
        await memory_store.log_tool_execution(
            tool_name="web_search", input_data={}, output="", success=True
        )

        executions = await memory_store.get_tool_executions(tool_name="bash")
        assert len(executions) == 1
        assert executions[0].tool_name == "bash"

    async def test_get_tool_executions_by_session(self, memory_store):
        session = await memory_store.get_or_create_session(
            provider="test", chat_id="chat-1", user_id="user-1"
        )
        await memory_store.log_tool_execution(
            tool_name="bash",
            input_data={},
            output="",
            success=True,
            session_id=session.id,
        )
        await memory_store.log_tool_execution(
            tool_name="bash", input_data={}, output="", success=True
        )

        executions = await memory_store.get_tool_executions(session_id=session.id)
        assert len(executions) == 1
