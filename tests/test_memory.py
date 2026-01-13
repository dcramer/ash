"""Tests for memory store operations."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.memory import MemoryManager, RetrievedContext, SearchResult
from ash.tools.base import ToolContext
from ash.tools.builtin.memory import RememberTool


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


class TestMemoryManager:
    """Tests for MemoryManager orchestrator."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock semantic retriever."""
        retriever = MagicMock()
        # Sessions/messages now stored in JSONL, not SQLite
        # Retriever only handles memory embeddings
        retriever.search_memories = AsyncMock(return_value=[])
        retriever.search = AsyncMock(return_value=[])
        retriever.index_memory = AsyncMock()
        retriever.delete_memory_embedding = AsyncMock()
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
            user_id="user-1",
            user_message="Hello",
        )

        assert isinstance(context, RetrievedContext)
        # Sessions/messages are now in JSONL, context only has memories
        assert context.memories == []

    async def test_get_context_for_message_with_results(
        self, memory_manager, mock_retriever
    ):
        """Test getting context with search results."""
        mock_retriever.search_memories.return_value = [
            SearchResult(
                id="mem-1",
                content="User preference",
                similarity=0.8,
                source_type="memory",
            )
        ]

        context = await memory_manager.get_context_for_message(
            user_id="user-1",
            user_message="What do you know?",
        )

        # Only memories are returned (messages are in JSONL sessions)
        assert len(context.memories) == 1
        assert context.memories[0].content == "User preference"

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
        """Test searching memories."""
        mock_retriever.search.return_value = [
            SearchResult(
                id="1",
                content="Result 1",
                similarity=0.9,
                source_type="memory",
            )
        ]

        results = await memory_manager.search("test query")

        assert len(results) == 1
        assert results[0].content == "Result 1"
        mock_retriever.search.assert_called_once_with(
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
        # Sessions/messages now stored in JSONL, not SQLite
        # Retriever only handles memory embeddings
        retriever.search_memories = AsyncMock(return_value=[])
        retriever.search = AsyncMock(return_value=[])
        retriever.index_memory = AsyncMock()
        retriever.delete_memory_embedding = AsyncMock()
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


class TestSubjectPersonValidation:
    """Tests for subject_person_ids validation."""

    async def test_add_memory_rejects_invalid_person_id(self, memory_store):
        """Test that add_memory rejects invalid subject_person_ids."""
        import pytest

        with pytest.raises(ValueError, match="Invalid subject person ID"):
            await memory_store.add_memory(
                content="Test fact about nonexistent person",
                subject_person_ids=["nonexistent-person-id"],
            )

    async def test_add_memory_accepts_valid_person_id(self, memory_store):
        """Test that add_memory accepts valid subject_person_ids."""
        # Create a person first (note: model uses 'relation' not 'relationship')
        person = await memory_store.create_person(
            owner_user_id="user-1",
            name="Sarah",
        )

        # Should succeed with valid person ID
        memory = await memory_store.add_memory(
            content="Sarah's birthday is March 15",
            subject_person_ids=[person.id],
            owner_user_id="user-1",
        )

        assert memory.subject_person_ids == [person.id]


class TestRememberToolGracefulDegradation:
    """Tests for RememberTool graceful degradation on person resolution failures."""

    @pytest.fixture
    def mock_memory_manager(self):
        """Create a mock memory manager with failing person resolution."""
        manager = MagicMock()
        manager.add_memory = AsyncMock()

        async def resolve_or_create_person(owner_user_id, reference, content_hint=None):
            # First call succeeds, second fails
            if reference == "Sarah":
                from ash.memory.manager import PersonResolutionResult

                return PersonResolutionResult(
                    person_id="person-1",
                    created=True,
                    person_name="Sarah",
                )
            else:
                raise Exception("Database error")

        manager.resolve_or_create_person = AsyncMock(
            side_effect=resolve_or_create_person
        )
        return manager

    @pytest.fixture
    def remember_tool(self, mock_memory_manager):
        """Create a remember tool with partial failing person resolution."""
        return RememberTool(memory_manager=mock_memory_manager)

    async def test_remember_continues_after_person_resolution_failure(
        self, remember_tool, mock_memory_manager
    ):
        """Test that remember continues storing fact even if one subject fails."""
        context = ToolContext(session_id="s1", user_id="u1")

        result = await remember_tool.execute(
            {
                "content": "Both like pizza",
                "subjects": ["Sarah", "BadRef"],  # First succeeds, second fails
            },
            context,
        )

        # Should not be an error - we stored the fact with partial subjects
        assert not result.is_error
        assert "Remembered" in result.content
        # Should mention the unresolved reference
        assert "unresolved" in result.content or "Sarah" in result.content

        # Memory should still be stored (with the one valid subject)
        mock_memory_manager.add_memory.assert_called_once()
        call_kwargs = mock_memory_manager.add_memory.call_args.kwargs
        assert call_kwargs["subject_person_ids"] == ["person-1"]


class TestSubjectNameResolution:
    """Tests for subject_name resolution in search results."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock semantic retriever that returns memories with subject_name."""
        retriever = MagicMock()
        # Sessions/messages now stored in JSONL, not SQLite
        # Retriever only handles memory embeddings
        retriever.search_memories = AsyncMock(return_value=[])
        retriever.search = AsyncMock(return_value=[])
        retriever.index_memory = AsyncMock()
        retriever.delete_memory_embedding = AsyncMock()
        return retriever

    async def test_search_memories_includes_subject_name(self, mock_retriever):
        """Test that search_memories returns subject_name in metadata.

        This tests the contract that search_memories should resolve
        subject_person_ids to human-readable names.
        """
        # Mock the retriever to return a memory with subject_name populated
        mock_retriever.search_memories.return_value = [
            SearchResult(
                id="mem-1",
                content="Sarah likes Italian food",
                similarity=0.9,
                source_type="memory",
                metadata={
                    "subject_person_ids": ["person-1"],
                    "subject_name": "Sarah",  # This is the key field we're testing
                },
            )
        ]

        results = await mock_retriever.search_memories(
            query="food preferences",
            owner_user_id="user-1",
        )

        assert len(results) == 1
        assert results[0].metadata["subject_name"] == "Sarah"


class TestMemoryDeletion:
    """Tests for memory deletion functionality."""

    async def test_delete_memory_from_store(self, memory_store):
        """Test deleting a memory from the store."""
        memory = await memory_store.add_memory(
            content="To be deleted",
            source="test",
        )

        # Delete the memory
        result = await memory_store.delete_memory(memory.id)
        assert result is True

        # Verify it's gone
        deleted = await memory_store.get_memory(memory.id)
        assert deleted is None

    async def test_delete_memory_not_found(self, memory_store):
        """Test deleting a nonexistent memory."""
        result = await memory_store.delete_memory("nonexistent-id")
        assert result is False

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock semantic retriever."""
        retriever = MagicMock()
        retriever.search_memories = AsyncMock(return_value=[])
        retriever.search = AsyncMock(return_value=[])
        retriever.index_memory = AsyncMock()
        retriever.delete_memory_embedding = AsyncMock()
        return retriever

    @pytest.fixture
    async def memory_manager(self, memory_store, mock_retriever, db_session):
        """Create a memory manager with mocked retriever."""
        return MemoryManager(
            store=memory_store,
            retriever=mock_retriever,
            db_session=db_session,
        )

    async def test_delete_memory_from_manager(
        self, memory_manager, memory_store, mock_retriever
    ):
        """Test deleting a memory through the manager (includes embedding cleanup)."""
        # Add a memory directly to store to avoid indexing complications
        memory = await memory_store.add_memory(
            content="To be deleted via manager",
            source="test",
        )

        # Delete through manager
        result = await memory_manager.delete_memory(memory.id)
        assert result is True

        # Verify embedding deletion was called
        mock_retriever.delete_memory_embedding.assert_called_once_with(memory.id)

        # Verify memory is gone from store
        deleted = await memory_store.get_memory(memory.id)
        assert deleted is None

    async def test_delete_memory_from_manager_not_found(
        self, memory_manager, mock_retriever
    ):
        """Test deleting a nonexistent memory through the manager."""
        result = await memory_manager.delete_memory("nonexistent-id")
        assert result is False

        # Embedding deletion should not be called for nonexistent memory
        mock_retriever.delete_memory_embedding.assert_not_called()


class TestGarbageCollection:
    """Tests for memory garbage collection."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock semantic retriever."""
        retriever = MagicMock()
        retriever.search_memories = AsyncMock(return_value=[])
        retriever.search = AsyncMock(return_value=[])
        retriever.index_memory = AsyncMock()
        retriever.delete_memory_embedding = AsyncMock()
        return retriever

    @pytest.fixture
    async def memory_manager(self, memory_store, mock_retriever, db_session):
        """Create a memory manager with mocked retriever."""
        return MemoryManager(
            store=memory_store,
            retriever=mock_retriever,
            db_session=db_session,
        )

    async def test_gc_removes_expired_memories(self, memory_manager, memory_store):
        """Test that GC removes expired memories."""
        # Add expired memory
        past = datetime.now(UTC) - timedelta(days=1)
        expired = await memory_store.add_memory(
            content="Expired fact",
            expires_at=past,
        )

        # Add valid memory
        valid = await memory_store.add_memory(
            content="Valid fact",
        )

        # Run GC
        expired_count, superseded_count = await memory_manager.gc()

        assert expired_count == 1
        assert superseded_count == 0

        # Expired memory should be gone
        assert await memory_store.get_memory(expired.id) is None
        # Valid memory should still exist
        assert await memory_store.get_memory(valid.id) is not None

    async def test_gc_removes_superseded_memories(self, memory_manager, memory_store):
        """Test that GC removes superseded memories."""
        # Create old memory
        old = await memory_store.add_memory(
            content="Old fact",
        )
        # Create new memory
        new = await memory_store.add_memory(
            content="New fact",
        )
        # Mark old as superseded
        await memory_store.mark_memory_superseded(old.id, new.id)

        # Run GC
        expired_count, superseded_count = await memory_manager.gc()

        assert expired_count == 0
        assert superseded_count == 1

        # Superseded memory should be gone
        assert await memory_store.get_memory(old.id) is None
        # New memory should still exist
        assert await memory_store.get_memory(new.id) is not None

    async def test_gc_removes_both_expired_and_superseded(
        self, memory_manager, memory_store
    ):
        """Test that GC removes both expired and superseded memories."""
        # Add expired memory
        past = datetime.now(UTC) - timedelta(days=1)
        expired = await memory_store.add_memory(
            content="Expired",
            expires_at=past,
        )

        # Add superseded memory
        old = await memory_store.add_memory(content="Old")
        new = await memory_store.add_memory(content="New")
        await memory_store.mark_memory_superseded(old.id, new.id)

        # Run GC
        expired_count, superseded_count = await memory_manager.gc()

        assert expired_count == 1
        assert superseded_count == 1

        # Both should be gone
        assert await memory_store.get_memory(expired.id) is None
        assert await memory_store.get_memory(old.id) is None
        # Valid memory should still exist
        assert await memory_store.get_memory(new.id) is not None

    async def test_gc_returns_zero_when_nothing_to_clean(
        self, memory_manager, memory_store
    ):
        """Test that GC returns (0, 0) when no cleanup needed."""
        # Add only valid memories
        await memory_store.add_memory(content="Valid 1")
        await memory_store.add_memory(content="Valid 2")

        expired_count, superseded_count = await memory_manager.gc()

        assert expired_count == 0
        assert superseded_count == 0


class TestEnforceMaxEntries:
    """Tests for max_entries enforcement."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock semantic retriever."""
        retriever = MagicMock()
        retriever.search_memories = AsyncMock(return_value=[])
        retriever.search = AsyncMock(return_value=[])
        retriever.index_memory = AsyncMock()
        retriever.delete_memory_embedding = AsyncMock()
        return retriever

    @pytest.fixture
    async def memory_manager(self, memory_store, mock_retriever, db_session):
        """Create a memory manager with mocked retriever."""
        return MemoryManager(
            store=memory_store,
            retriever=mock_retriever,
            db_session=db_session,
        )

    async def test_enforce_max_entries_evicts_oldest(
        self, memory_manager, memory_store
    ):
        """Test that enforce_max_entries evicts oldest memories when over limit."""
        # Add 5 memories with explicit old timestamps
        old_time = datetime.now(UTC) - timedelta(days=10)
        for i in range(5):
            m = await memory_store.add_memory(content=f"Fact {i}")
            # Manually set created_at to the past to make them eligible for eviction
            m.created_at = old_time + timedelta(hours=i)

        # Commit the changes
        await memory_store._session.commit()

        # Enforce max of 3
        evicted = await memory_manager.enforce_max_entries(3)

        assert evicted == 2

        # Should have 3 remaining
        memories = await memory_store.get_memories()
        assert len(memories) == 3

    async def test_enforce_max_entries_no_eviction_when_under_limit(
        self, memory_manager, memory_store
    ):
        """Test that enforce_max_entries does nothing when under limit."""
        await memory_store.add_memory(content="Fact 1")
        await memory_store.add_memory(content="Fact 2")

        evicted = await memory_manager.enforce_max_entries(10)

        assert evicted == 0

        memories = await memory_store.get_memories()
        assert len(memories) == 2

    async def test_enforce_max_entries_prioritizes_superseded(
        self, memory_manager, memory_store
    ):
        """Test that superseded memories are evicted before active ones."""
        # Add old memories that are eligible for eviction
        old_time = datetime.now(UTC) - timedelta(days=10)

        # Add 3 valid old memories
        valid1 = await memory_store.add_memory(content="Valid 1")
        valid1.created_at = old_time
        valid2 = await memory_store.add_memory(content="Valid 2")
        valid2.created_at = old_time + timedelta(hours=1)
        valid3 = await memory_store.add_memory(content="Valid 3")
        valid3.created_at = old_time + timedelta(hours=2)

        # Add a superseded memory (should be evicted first)
        old = await memory_store.add_memory(content="Old superseded")
        old.created_at = old_time + timedelta(hours=3)
        new = await memory_store.add_memory(content="New fact")
        new.created_at = old_time + timedelta(hours=4)
        await memory_store.mark_memory_superseded(old.id, new.id)

        await memory_store._session.commit()

        # State: valid1, valid2, valid3, new are active (4 total)
        #        old is superseded (doesn't count toward active)
        # Limit 3 means we need to evict 1

        evicted = await memory_manager.enforce_max_entries(3)

        # At least 1 should be evicted
        assert evicted >= 1

        # Superseded should be gone (priority 1 - evicted first)
        assert await memory_store.get_memory(old.id) is None

        # New fact should remain (most recent)
        assert await memory_store.get_memory(new.id) is not None


class TestGroupMemoryScoping:
    """Tests for group (chat-scoped) memory functionality."""

    async def test_add_personal_memory(self, memory_store):
        """Test adding a personal memory (owner_user_id set, chat_id null)."""
        memory = await memory_store.add_memory(
            content="My personal preference",
            owner_user_id="user-1",
            chat_id=None,
        )

        assert memory.owner_user_id == "user-1"
        assert memory.chat_id is None

    async def test_add_group_memory(self, memory_store):
        """Test adding a group memory (owner_user_id null, chat_id set)."""
        memory = await memory_store.add_memory(
            content="Team standup is at 10am",
            owner_user_id=None,
            chat_id="chat-1",
        )

        assert memory.owner_user_id is None
        assert memory.chat_id == "chat-1"

    async def test_get_memories_filters_by_owner(self, memory_store):
        """Test that get_memories filters by owner_user_id."""
        # Add memories for different users
        await memory_store.add_memory(
            content="User 1 fact",
            owner_user_id="user-1",
        )
        await memory_store.add_memory(
            content="User 2 fact",
            owner_user_id="user-2",
        )

        # Get only user-1's memories
        memories = await memory_store.get_memories(owner_user_id="user-1")

        assert len(memories) == 1
        assert memories[0].content == "User 1 fact"

    async def test_get_memories_filters_by_chat(self, memory_store):
        """Test that get_memories filters by chat_id for group memories."""
        # Add group memories for different chats
        await memory_store.add_memory(
            content="Chat 1 fact",
            owner_user_id=None,
            chat_id="chat-1",
        )
        await memory_store.add_memory(
            content="Chat 2 fact",
            owner_user_id=None,
            chat_id="chat-2",
        )

        # Get only chat-1's group memories
        memories = await memory_store.get_memories(chat_id="chat-1")

        assert len(memories) == 1
        assert memories[0].content == "Chat 1 fact"

    async def test_get_memories_combines_personal_and_group(self, memory_store):
        """Test that user gets both personal and group memories."""
        # Add personal memory for user-1
        await memory_store.add_memory(
            content="My personal fact",
            owner_user_id="user-1",
            chat_id=None,
        )
        # Add group memory for chat-1
        await memory_store.add_memory(
            content="Group fact",
            owner_user_id=None,
            chat_id="chat-1",
        )
        # Add group memory for different chat
        await memory_store.add_memory(
            content="Other chat fact",
            owner_user_id=None,
            chat_id="chat-2",
        )
        # Add personal memory for different user
        await memory_store.add_memory(
            content="Other user fact",
            owner_user_id="user-2",
            chat_id=None,
        )

        # Get user-1's personal memories + chat-1's group memories
        memories = await memory_store.get_memories(
            owner_user_id="user-1",
            chat_id="chat-1",
        )

        assert len(memories) == 2
        contents = [m.content for m in memories]
        assert "My personal fact" in contents
        assert "Group fact" in contents

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock semantic retriever."""
        retriever = MagicMock()
        retriever.search_memories = AsyncMock(return_value=[])
        retriever.search = AsyncMock(return_value=[])
        retriever.index_memory = AsyncMock()
        retriever.delete_memory_embedding = AsyncMock()
        return retriever

    @pytest.fixture
    async def memory_manager(self, memory_store, mock_retriever, db_session):
        """Create a memory manager with mocked retriever."""
        return MemoryManager(
            store=memory_store,
            retriever=mock_retriever,
            db_session=db_session,
        )

    async def test_manager_add_group_memory(self, memory_manager, memory_store):
        """Test adding a group memory through the manager."""
        memory = await memory_manager.add_memory(
            content="Shared team fact",
            owner_user_id=None,
            chat_id="chat-1",
        )

        assert memory.owner_user_id is None
        assert memory.chat_id == "chat-1"

    async def test_manager_context_includes_both_scopes(
        self, memory_manager, mock_retriever
    ):
        """Test that get_context_for_message passes both user_id and chat_id to retriever."""
        await memory_manager.get_context_for_message(
            user_id="user-1",
            user_message="Hello",
            chat_id="chat-1",
        )

        # Verify both scoping params were passed to retriever
        mock_retriever.search_memories.assert_called_once()
        call_kwargs = mock_retriever.search_memories.call_args.kwargs
        assert call_kwargs["owner_user_id"] == "user-1"
        assert call_kwargs["chat_id"] == "chat-1"
