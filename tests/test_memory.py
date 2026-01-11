"""Tests for memory store operations."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.memory.manager import MemoryManager, RetrievedContext
from ash.memory.retrieval import SearchResult
from ash.tools.base import ToolContext
from ash.tools.builtin.memory import RecallTool, RememberTool


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


class TestMemoryOperations:
    """Tests for memory entry operations."""

    async def test_add_memory(self, memory_store):
        memory = await memory_store.add_memory(
            content="Python is a programming language.",
            source="manual",
        )
        assert memory.id is not None
        assert memory.content == "Python is a programming language."
        assert memory.source == "manual"

    async def test_add_memory_with_expiry(self, memory_store):
        expires = datetime.now(UTC) + timedelta(days=7)
        memory = await memory_store.add_memory(
            content="Temporary memory",
            expires_at=expires,
        )
        assert memory.expires_at == expires

    async def test_get_memories(self, memory_store):
        await memory_store.add_memory(content="Fact 1")
        await memory_store.add_memory(content="Fact 2")

        memories = await memory_store.get_memories()
        assert len(memories) == 2

    async def test_get_memories_excludes_expired(self, memory_store):
        # Add expired memory
        past = datetime.now(UTC) - timedelta(days=1)
        await memory_store.add_memory(
            content="Expired fact",
            expires_at=past,
        )
        # Add valid memory
        await memory_store.add_memory(content="Valid fact")

        memories = await memory_store.get_memories(include_expired=False)
        assert len(memories) == 1
        assert memories[0].content == "Valid fact"

    async def test_get_memories_includes_expired(self, memory_store):
        past = datetime.now(UTC) - timedelta(days=1)
        await memory_store.add_memory(content="Expired", expires_at=past)
        await memory_store.add_memory(content="Valid")

        memories = await memory_store.get_memories(include_expired=True)
        assert len(memories) == 2


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


class TestMemoryManager:
    """Tests for MemoryManager orchestrator."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock semantic retriever."""
        retriever = MagicMock()
        retriever.search_messages = AsyncMock(return_value=[])
        retriever.search_memories = AsyncMock(return_value=[])
        retriever.search_all = AsyncMock(return_value=[])
        retriever.index_message = AsyncMock()
        retriever.index_memory = AsyncMock()
        return retriever

    @pytest.fixture
    async def memory_manager(self, memory_store, mock_retriever, db_session):
        """Create a memory manager with mocked retriever."""
        return MemoryManager(
            store=memory_store,
            retriever=mock_retriever,
            db_session=db_session,
        )

    async def test_get_context_for_message_empty(self, memory_manager):
        """Test getting context when no relevant data exists."""
        context = await memory_manager.get_context_for_message(
            session_id="session-1",
            user_id="user-1",
            user_message="Hello",
        )

        assert isinstance(context, RetrievedContext)
        assert context.messages == []
        assert context.memories == []

    async def test_get_context_for_message_with_results(
        self, memory_manager, mock_retriever
    ):
        """Test getting context with search results."""
        mock_retriever.search_messages.return_value = [
            SearchResult(
                id="msg-1",
                content="Previous conversation",
                similarity=0.9,
                source_type="message",
            )
        ]
        mock_retriever.search_memories.return_value = [
            SearchResult(
                id="mem-1",
                content="User preference",
                similarity=0.8,
                source_type="memory",
            )
        ]

        context = await memory_manager.get_context_for_message(
            session_id="session-1",
            user_id="user-1",
            user_message="What do you know?",
        )

        assert len(context.messages) == 1
        assert context.messages[0].content == "Previous conversation"
        assert len(context.memories) == 1
        assert context.memories[0].content == "User preference"

    async def test_persist_turn(self, memory_manager, memory_store, mock_retriever):
        """Test persisting a conversation turn."""
        # Create session first
        session = await memory_store.get_or_create_session(
            provider="test",
            chat_id="chat-1",
            user_id="user-1",
        )

        await memory_manager.persist_turn(
            session_id=session.id,
            user_message="Hello there",
            assistant_response="Hi! How can I help?",
        )

        # Check messages were stored
        messages = await memory_store.get_messages(session.id)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello there"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi! How can I help?"

        # Check indexing was called
        assert mock_retriever.index_message.call_count == 2

    async def test_add_memory(self, memory_manager, memory_store, mock_retriever):
        """Test adding memory entry."""
        memory = await memory_manager.add_memory(
            content="User likes Python",
            source="remember_tool",
        )

        assert memory.content == "User likes Python"
        assert memory.source == "remember_tool"

        # Check indexing was called
        mock_retriever.index_memory.assert_called_once()

    async def test_add_memory_with_expiration(self, memory_manager):
        """Test adding memory with expiration."""
        memory = await memory_manager.add_memory(
            content="Temporary fact",
            expires_in_days=7,
        )

        assert memory.expires_at is not None
        assert memory.expires_at > datetime.now(UTC)

    async def test_search(self, memory_manager, mock_retriever):
        """Test searching all memory."""
        mock_retriever.search_all.return_value = [
            SearchResult(
                id="1",
                content="Result 1",
                similarity=0.9,
                source_type="knowledge",
            )
        ]

        results = await memory_manager.search("test query")

        assert len(results) == 1
        assert results[0].content == "Result 1"
        mock_retriever.search_all.assert_called_once_with(
            "test query",
            limit=5,
            subject_person_id=None,
            owner_user_id=None,
            chat_id=None,
        )


class TestRememberTool:
    """Tests for the remember tool."""

    @pytest.fixture
    def mock_memory_manager(self):
        """Create a mock memory manager."""
        manager = MagicMock()
        manager.add_memory = AsyncMock()
        return manager

    @pytest.fixture
    def remember_tool(self, mock_memory_manager):
        """Create a remember tool with mocked manager."""
        return RememberTool(memory_manager=mock_memory_manager)

    async def test_remember_stores_content(self, remember_tool, mock_memory_manager):
        """Test that remember tool stores content."""
        context = ToolContext(session_id="s1", user_id="u1")
        result = await remember_tool.execute(
            {"content": "User prefers dark mode"},
            context,
        )

        assert not result.is_error
        assert "Remembered" in result.content
        mock_memory_manager.add_memory.assert_called_once_with(
            content="User prefers dark mode",
            source="remember_tool",
            expires_in_days=None,
            owner_user_id="u1",
            chat_id=None,
            subject_person_ids=None,
        )

    async def test_remember_with_expiration(self, remember_tool, mock_memory_manager):
        """Test remembering with expiration."""
        context = ToolContext()
        await remember_tool.execute(
            {"content": "Temporary note", "expires_in_days": 30},
            context,
        )

        mock_memory_manager.add_memory.assert_called_once_with(
            content="Temporary note",
            source="remember_tool",
            expires_in_days=30,
            owner_user_id=None,
            chat_id=None,
            subject_person_ids=None,
        )

    async def test_remember_missing_content(self, remember_tool):
        """Test error when content is missing."""
        context = ToolContext()
        result = await remember_tool.execute({}, context)

        assert result.is_error
        assert "Missing required parameter" in result.content

    async def test_remember_handles_error(self, remember_tool, mock_memory_manager):
        """Test error handling when storage fails."""
        mock_memory_manager.add_memory.side_effect = Exception("DB error")
        context = ToolContext()

        result = await remember_tool.execute(
            {"content": "Test"},
            context,
        )

        assert result.is_error
        assert "Failed to store" in result.content
        assert "DB error" in result.content


class TestMemorySupersession:
    """Tests for memory supersession functionality."""

    async def test_mark_memory_superseded(self, memory_store):
        """Test marking a memory as superseded."""
        # Create old memory
        old_memory = await memory_store.add_memory(
            content="User's favorite color is red",
            owner_user_id="user-1",
        )
        # Create new memory
        new_memory = await memory_store.add_memory(
            content="User's favorite color is blue",
            owner_user_id="user-1",
        )

        # Mark old as superseded
        result = await memory_store.mark_memory_superseded(
            memory_id=old_memory.id,
            superseded_by_id=new_memory.id,
        )

        assert result is True

        # Verify the old memory is updated
        old_refreshed = await memory_store.get_memory(old_memory.id)
        assert old_refreshed.superseded_at is not None
        assert old_refreshed.superseded_by_id == new_memory.id

    async def test_mark_memory_superseded_not_found(self, memory_store):
        """Test marking nonexistent memory as superseded."""
        result = await memory_store.mark_memory_superseded(
            memory_id="nonexistent-id",
            superseded_by_id="some-id",
        )
        assert result is False

    async def test_get_memories_excludes_superseded(self, memory_store):
        """Test that get_memories excludes superseded memories by default."""
        # Create old memory
        old_memory = await memory_store.add_memory(
            content="Old fact",
            owner_user_id="user-1",
        )
        # Create new memory
        new_memory = await memory_store.add_memory(
            content="New fact",
            owner_user_id="user-1",
        )
        # Supersede old memory
        await memory_store.mark_memory_superseded(
            memory_id=old_memory.id,
            superseded_by_id=new_memory.id,
        )

        # Default: should only get the new memory
        memories = await memory_store.get_memories(include_superseded=False)
        assert len(memories) == 1
        assert memories[0].content == "New fact"

    async def test_get_memories_includes_superseded(self, memory_store):
        """Test that get_memories can include superseded memories."""
        old_memory = await memory_store.add_memory(content="Old fact")
        new_memory = await memory_store.add_memory(content="New fact")
        await memory_store.mark_memory_superseded(
            memory_id=old_memory.id,
            superseded_by_id=new_memory.id,
        )

        # With include_superseded=True: should get both
        memories = await memory_store.get_memories(include_superseded=True)
        assert len(memories) == 2


class TestMemoryManagerSupersession:
    """Tests for MemoryManager supersession logic."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock semantic retriever."""
        retriever = MagicMock()
        retriever.search_messages = AsyncMock(return_value=[])
        retriever.search_memories = AsyncMock(return_value=[])
        retriever.search_all = AsyncMock(return_value=[])
        retriever.index_message = AsyncMock()
        retriever.index_memory = AsyncMock()
        return retriever

    @pytest.fixture
    async def memory_manager(self, memory_store, mock_retriever, db_session):
        """Create a memory manager with mocked retriever."""
        return MemoryManager(
            store=memory_store,
            retriever=mock_retriever,
            db_session=db_session,
        )

    async def test_add_memory_supersedes_conflicting(
        self, memory_manager, memory_store, mock_retriever
    ):
        """Test that adding a memory supersedes conflicting ones."""
        # Create first memory
        old_memory = await memory_store.add_memory(
            content="User's favorite color is red",
            owner_user_id="user-1",
        )

        # Mock the retriever to return the old memory as a conflict
        mock_retriever.search_memories.return_value = [
            SearchResult(
                id=old_memory.id,
                content=old_memory.content,
                similarity=0.85,  # Above 0.75 threshold
                source_type="memory",
                metadata={"subject_person_id": None},
            )
        ]

        # Add new conflicting memory via manager (which triggers supersession)
        new_memory = await memory_manager.add_memory(
            content="User's favorite color is blue",
            owner_user_id="user-1",
        )

        # Verify old memory was superseded
        old_refreshed = await memory_store.get_memory(old_memory.id)
        assert old_refreshed.superseded_at is not None
        assert old_refreshed.superseded_by_id == new_memory.id

    async def test_add_memory_no_conflict_below_threshold(
        self, memory_manager, memory_store, mock_retriever
    ):
        """Test that memories below threshold are not superseded."""
        old_memory = await memory_store.add_memory(
            content="User likes pizza",
            owner_user_id="user-1",
        )

        # Mock retriever to return the old memory with low similarity
        mock_retriever.search_memories.return_value = [
            SearchResult(
                id=old_memory.id,
                content=old_memory.content,
                similarity=0.5,  # Below 0.75 threshold
                source_type="memory",
                metadata={},
            )
        ]

        # Add unrelated memory
        await memory_manager.add_memory(
            content="User likes coffee",
            owner_user_id="user-1",
        )

        # Old memory should NOT be superseded
        old_refreshed = await memory_store.get_memory(old_memory.id)
        assert old_refreshed.superseded_at is None

    async def test_find_conflicting_memories_filters_by_subject(
        self, memory_manager, mock_retriever
    ):
        """Test that conflict detection respects subject filtering."""
        # Mock retriever to return memories about different subjects
        mock_retriever.search_memories.return_value = [
            SearchResult(
                id="mem-1",
                content="Sarah likes pizza",
                similarity=0.9,
                source_type="memory",
                metadata={"subject_person_ids": ["person-1"]},  # About Sarah
            ),
            SearchResult(
                id="mem-2",
                content="Michael likes sushi",
                similarity=0.85,
                source_type="memory",
                metadata={"subject_person_ids": ["person-2"]},  # About Michael
            ),
        ]

        # Find conflicts for memories about Sarah (person-1)
        conflicts = await memory_manager.find_conflicting_memories(
            new_content="Sarah likes pasta",
            owner_user_id="user-1",
            subject_person_ids=["person-1"],
        )

        # Only memory about Sarah should be a conflict
        assert len(conflicts) == 1
        assert conflicts[0][0] == "mem-1"

    async def test_subjectless_memory_does_not_supersede_subject_memory(
        self,
        memory_manager,
        mock_retriever,
    ):
        """Test that memories without subjects don't supersede person-specific memories."""
        # Mock retriever to return memories WITH subjects
        mock_retriever.search_memories.return_value = [
            SearchResult(
                id="mem-1",
                content="Sarah likes pizza",
                similarity=0.9,
                source_type="memory",
                metadata={"subject_person_ids": ["person-1"]},  # About Sarah
            ),
            SearchResult(
                id="mem-2",
                content="General food preferences",
                similarity=0.85,
                source_type="memory",
                metadata={"subject_person_ids": None},  # No subject
            ),
        ]

        # Find conflicts for a memory with NO subjects
        conflicts = await memory_manager.find_conflicting_memories(
            new_content="Family likes pizza",
            owner_user_id="user-1",
            subject_person_ids=None,  # No subjects
        )

        # Only the subjectless memory should be a conflict
        assert len(conflicts) == 1
        assert conflicts[0][0] == "mem-2"


class TestRecallTool:
    """Tests for the recall tool."""

    @pytest.fixture
    def mock_memory_manager(self):
        """Create a mock memory manager."""
        manager = MagicMock()
        manager.search = AsyncMock(return_value=[])
        return manager

    @pytest.fixture
    def recall_tool(self, mock_memory_manager):
        """Create a recall tool with mocked manager."""
        return RecallTool(memory_manager=mock_memory_manager)

    async def test_recall_searches_memory(self, recall_tool, mock_memory_manager):
        """Test that recall tool searches memory."""
        mock_memory_manager.search.return_value = [
            SearchResult(
                id="1",
                content="User likes Python",
                similarity=0.9,
                source_type="memory",
            ),
            SearchResult(
                id="2",
                content="Previous discussion about coding",
                similarity=0.8,
                source_type="message",
            ),
        ]
        context = ToolContext()
        result = await recall_tool.execute({"query": "python"}, context)

        assert not result.is_error
        assert "Found relevant memories" in result.content
        assert "User likes Python" in result.content
        assert "[memory]" in result.content
        assert "[message]" in result.content

    async def test_recall_no_results(self, recall_tool, mock_memory_manager):
        """Test recall when no memories found."""
        context = ToolContext()
        result = await recall_tool.execute({"query": "unknown"}, context)

        assert not result.is_error
        assert "No relevant memories found" in result.content

    async def test_recall_missing_query(self, recall_tool):
        """Test error when query is missing."""
        context = ToolContext()
        result = await recall_tool.execute({}, context)

        assert result.is_error
        assert "Missing required parameter" in result.content

    async def test_recall_handles_error(self, recall_tool, mock_memory_manager):
        """Test error handling when search fails."""
        mock_memory_manager.search.side_effect = Exception("Search error")
        context = ToolContext()

        result = await recall_tool.execute({"query": "test"}, context)

        assert result.is_error
        assert "Failed to search memory" in result.content
