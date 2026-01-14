"""Tests for memory system behavior.

Tests focus on:
- Supersession logic (core business behavior)
- Garbage collection and eviction
- Scoping rules (personal vs group)
- Error handling
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.memory import MemoryManager, SearchResult


@pytest.fixture
def mock_retriever():
    """Create a mock semantic retriever."""
    retriever = MagicMock()
    retriever.search_memories = AsyncMock(return_value=[])
    retriever.search = AsyncMock(return_value=[])
    retriever.index_memory = AsyncMock()
    retriever.delete_memory_embedding = AsyncMock()
    return retriever


@pytest.fixture
async def memory_manager(memory_store, mock_retriever, db_session):
    """Create a memory manager with mocked retriever."""
    return MemoryManager(
        store=memory_store,
        retriever=mock_retriever,
        db_session=db_session,
    )


class TestMemorySupersession:
    """Tests for memory supersession - core business logic."""

    async def test_mark_memory_superseded(self, memory_store):
        """Test marking a memory as superseded."""
        old_memory = await memory_store.add_memory(
            content="User's favorite color is red",
            owner_user_id="user-1",
        )
        new_memory = await memory_store.add_memory(
            content="User's favorite color is blue",
            owner_user_id="user-1",
        )

        result = await memory_store.mark_memory_superseded(
            memory_id=old_memory.id,
            superseded_by_id=new_memory.id,
        )

        assert result is True
        old_refreshed = await memory_store.get_memory(old_memory.id)
        assert old_refreshed.superseded_at is not None
        assert old_refreshed.superseded_by_id == new_memory.id

    async def test_get_memories_excludes_superseded_by_default(self, memory_store):
        """Test that get_memories excludes superseded memories."""
        old_memory = await memory_store.add_memory(content="Old fact")
        new_memory = await memory_store.add_memory(content="New fact")
        await memory_store.mark_memory_superseded(old_memory.id, new_memory.id)

        memories = await memory_store.get_memories(include_superseded=False)

        assert len(memories) == 1
        assert memories[0].content == "New fact"

    async def test_get_memories_can_include_superseded(self, memory_store):
        """Test that get_memories can include superseded memories."""
        old_memory = await memory_store.add_memory(content="Old fact")
        new_memory = await memory_store.add_memory(content="New fact")
        await memory_store.mark_memory_superseded(old_memory.id, new_memory.id)

        memories = await memory_store.get_memories(include_superseded=True)

        assert len(memories) == 2


class TestMemoryManagerSupersession:
    """Tests for automatic supersession via MemoryManager."""

    async def test_add_memory_supersedes_conflicting(
        self, memory_manager, memory_store, mock_retriever
    ):
        """Test that adding a memory supersedes high-similarity conflicts."""
        old_memory = await memory_store.add_memory(
            content="User's favorite color is red",
            owner_user_id="user-1",
        )

        mock_retriever.search_memories.return_value = [
            SearchResult(
                id=old_memory.id,
                content=old_memory.content,
                similarity=0.85,  # Above 0.75 threshold
                source_type="memory",
                metadata={"subject_person_ids": None},
            )
        ]

        new_memory = await memory_manager.add_memory(
            content="User's favorite color is blue",
            owner_user_id="user-1",
        )

        old_refreshed = await memory_store.get_memory(old_memory.id)
        assert old_refreshed.superseded_at is not None
        assert old_refreshed.superseded_by_id == new_memory.id

    async def test_no_supersession_below_threshold(
        self, memory_manager, memory_store, mock_retriever
    ):
        """Test that memories below similarity threshold are not superseded."""
        old_memory = await memory_store.add_memory(
            content="User likes pizza",
            owner_user_id="user-1",
        )

        mock_retriever.search_memories.return_value = [
            SearchResult(
                id=old_memory.id,
                content=old_memory.content,
                similarity=0.5,  # Below 0.75 threshold
                source_type="memory",
                metadata={},
            )
        ]

        await memory_manager.add_memory(
            content="User likes coffee",
            owner_user_id="user-1",
        )

        old_refreshed = await memory_store.get_memory(old_memory.id)
        assert old_refreshed.superseded_at is None

    async def test_conflict_detection_respects_subject_filtering(
        self, memory_manager, mock_retriever
    ):
        """Test that conflict detection filters by subject."""
        mock_retriever.search_memories.return_value = [
            SearchResult(
                id="mem-1",
                content="Sarah likes pizza",
                similarity=0.9,
                source_type="memory",
                metadata={"subject_person_ids": ["person-1"]},
            ),
            SearchResult(
                id="mem-2",
                content="Michael likes sushi",
                similarity=0.85,
                source_type="memory",
                metadata={"subject_person_ids": ["person-2"]},
            ),
        ]

        conflicts = await memory_manager.find_conflicting_memories(
            new_content="Sarah likes pasta",
            owner_user_id="user-1",
            subject_person_ids=["person-1"],
        )

        # Only memory about Sarah should be a conflict
        assert len(conflicts) == 1
        assert conflicts[0][0] == "mem-1"

    async def test_subjectless_memory_does_not_supersede_subject_memory(
        self, memory_manager, mock_retriever
    ):
        """Test that general facts don't supersede person-specific facts."""
        mock_retriever.search_memories.return_value = [
            SearchResult(
                id="mem-1",
                content="Sarah likes pizza",
                similarity=0.9,
                source_type="memory",
                metadata={"subject_person_ids": ["person-1"]},
            ),
            SearchResult(
                id="mem-2",
                content="General food preferences",
                similarity=0.85,
                source_type="memory",
                metadata={"subject_person_ids": None},
            ),
        ]

        conflicts = await memory_manager.find_conflicting_memories(
            new_content="Family likes pizza",
            owner_user_id="user-1",
            subject_person_ids=None,
        )

        # Only the subjectless memory should conflict
        assert len(conflicts) == 1
        assert conflicts[0][0] == "mem-2"


class TestGarbageCollection:
    """Tests for memory garbage collection."""

    async def test_gc_removes_expired_memories(self, memory_manager, memory_store):
        """Test that GC removes expired memories."""
        past = datetime.now(UTC) - timedelta(days=1)
        expired = await memory_store.add_memory(content="Expired", expires_at=past)
        valid = await memory_store.add_memory(content="Valid")

        expired_count, superseded_count = await memory_manager.gc()

        assert expired_count == 1
        assert superseded_count == 0
        assert await memory_store.get_memory(expired.id) is None
        assert await memory_store.get_memory(valid.id) is not None

    async def test_gc_removes_superseded_memories(self, memory_manager, memory_store):
        """Test that GC removes superseded memories."""
        old = await memory_store.add_memory(content="Old fact")
        new = await memory_store.add_memory(content="New fact")
        await memory_store.mark_memory_superseded(old.id, new.id)

        expired_count, superseded_count = await memory_manager.gc()

        assert expired_count == 0
        assert superseded_count == 1
        assert await memory_store.get_memory(old.id) is None
        assert await memory_store.get_memory(new.id) is not None

    async def test_gc_handles_empty_state(self, memory_manager, memory_store):
        """Test that GC works with only valid memories."""
        await memory_store.add_memory(content="Valid 1")
        await memory_store.add_memory(content="Valid 2")

        expired_count, superseded_count = await memory_manager.gc()

        assert expired_count == 0
        assert superseded_count == 0


class TestEnforceMaxEntries:
    """Tests for max_entries eviction policy."""

    async def test_evicts_oldest_when_over_limit(self, memory_manager, memory_store):
        """Test that enforce_max_entries evicts oldest memories."""
        old_time = datetime.now(UTC) - timedelta(days=10)
        for i in range(5):
            m = await memory_store.add_memory(content=f"Fact {i}")
            m.created_at = old_time + timedelta(hours=i)
        await memory_store._session.commit()

        evicted = await memory_manager.enforce_max_entries(3)

        assert evicted == 2
        memories = await memory_store.get_memories()
        assert len(memories) == 3

    async def test_no_eviction_when_under_limit(self, memory_manager, memory_store):
        """Test that no eviction happens when under limit."""
        await memory_store.add_memory(content="Fact 1")
        await memory_store.add_memory(content="Fact 2")

        evicted = await memory_manager.enforce_max_entries(10)

        assert evicted == 0

    async def test_prioritizes_superseded_for_eviction(
        self, memory_manager, memory_store
    ):
        """Test that superseded memories are evicted first."""
        old_time = datetime.now(UTC) - timedelta(days=10)

        valid1 = await memory_store.add_memory(content="Valid 1")
        valid1.created_at = old_time
        valid2 = await memory_store.add_memory(content="Valid 2")
        valid2.created_at = old_time + timedelta(hours=1)
        valid3 = await memory_store.add_memory(content="Valid 3")
        valid3.created_at = old_time + timedelta(hours=2)

        old = await memory_store.add_memory(content="Old superseded")
        old.created_at = old_time + timedelta(hours=3)
        new = await memory_store.add_memory(content="New fact")
        new.created_at = old_time + timedelta(hours=4)
        await memory_store.mark_memory_superseded(old.id, new.id)
        await memory_store._session.commit()

        evicted = await memory_manager.enforce_max_entries(3)

        assert evicted >= 1
        # Superseded should be gone first
        assert await memory_store.get_memory(old.id) is None
        # New fact should remain
        assert await memory_store.get_memory(new.id) is not None


class TestMemoryScoping:
    """Tests for personal vs group memory scoping."""

    async def test_personal_memory_has_owner_no_chat(self, memory_store):
        """Test personal memory scoping."""
        memory = await memory_store.add_memory(
            content="My personal preference",
            owner_user_id="user-1",
            chat_id=None,
        )

        assert memory.owner_user_id == "user-1"
        assert memory.chat_id is None

    async def test_group_memory_has_chat_no_owner(self, memory_store):
        """Test group memory scoping."""
        memory = await memory_store.add_memory(
            content="Team standup is at 10am",
            owner_user_id=None,
            chat_id="chat-1",
        )

        assert memory.owner_user_id is None
        assert memory.chat_id == "chat-1"

    async def test_get_memories_combines_personal_and_group(self, memory_store):
        """Test that user gets both personal and group memories."""
        await memory_store.add_memory(
            content="My personal fact", owner_user_id="user-1"
        )
        await memory_store.add_memory(
            content="Group fact", owner_user_id=None, chat_id="chat-1"
        )
        await memory_store.add_memory(
            content="Other chat fact", owner_user_id=None, chat_id="chat-2"
        )
        await memory_store.add_memory(content="Other user fact", owner_user_id="user-2")

        memories = await memory_store.get_memories(
            owner_user_id="user-1", chat_id="chat-1"
        )

        assert len(memories) == 2
        contents = [m.content for m in memories]
        assert "My personal fact" in contents
        assert "Group fact" in contents


class TestSubjectValidation:
    """Tests for subject person validation."""

    async def test_rejects_invalid_person_id(self, memory_store):
        """Test that invalid person IDs are rejected."""
        with pytest.raises(ValueError, match="Invalid subject person ID"):
            await memory_store.add_memory(
                content="Test fact about nonexistent person",
                subject_person_ids=["nonexistent-id"],
            )

    async def test_accepts_valid_person_id(self, memory_store):
        """Test that valid person IDs are accepted."""
        person = await memory_store.create_person(owner_user_id="user-1", name="Sarah")

        memory = await memory_store.add_memory(
            content="Sarah's birthday is March 15",
            subject_person_ids=[person.id],
            owner_user_id="user-1",
        )

        assert memory.subject_person_ids == [person.id]


class TestMemoryDeletion:
    """Tests for memory deletion."""

    async def test_delete_removes_memory(self, memory_store):
        """Test that deletion removes the memory."""
        memory = await memory_store.add_memory(content="To be deleted")

        result = await memory_store.delete_memory(memory.id)

        assert result is True
        assert await memory_store.get_memory(memory.id) is None

    async def test_delete_nonexistent_returns_false(self, memory_store):
        """Test that deleting nonexistent memory returns False."""
        result = await memory_store.delete_memory("nonexistent-id")
        assert result is False


class TestMemoryAuthorization:
    """Tests for best-effort authorization checks."""

    async def test_delete_own_memory_succeeds(self, memory_store):
        """Test that user can delete their own memory."""
        memory = await memory_store.add_memory(
            content="My secret", owner_user_id="user-1"
        )

        result = await memory_store.delete_memory(memory.id, owner_user_id="user-1")

        assert result is True
        assert await memory_store.get_memory(memory.id) is None

    async def test_delete_other_user_memory_fails(self, memory_store):
        """Test that user cannot delete another user's memory."""
        memory = await memory_store.add_memory(
            content="User 1 secret", owner_user_id="user-1"
        )

        result = await memory_store.delete_memory(memory.id, owner_user_id="user-2")

        assert result is False
        assert await memory_store.get_memory(memory.id) is not None

    async def test_delete_group_memory_with_chat_context(self, memory_store):
        """Test that chat member can delete group memory."""
        memory = await memory_store.add_memory(
            content="Group fact", owner_user_id=None, chat_id="chat-1"
        )

        result = await memory_store.delete_memory(memory.id, chat_id="chat-1")

        assert result is True

    async def test_delete_group_memory_wrong_chat_fails(self, memory_store):
        """Test that user in different chat cannot delete group memory."""
        memory = await memory_store.add_memory(
            content="Group fact", owner_user_id=None, chat_id="chat-1"
        )

        result = await memory_store.delete_memory(memory.id, chat_id="chat-2")

        assert result is False
        assert await memory_store.get_memory(memory.id) is not None

    async def test_get_person_with_wrong_owner_returns_none(self, memory_store):
        """Test that get_person returns None for non-owner."""
        person = await memory_store.create_person(owner_user_id="user-1", name="Sarah")

        result = await memory_store.get_person(person.id, owner_user_id="user-2")

        assert result is None

    async def test_get_person_with_correct_owner_succeeds(self, memory_store):
        """Test that get_person returns person for owner."""
        person = await memory_store.create_person(owner_user_id="user-1", name="Sarah")

        result = await memory_store.get_person(person.id, owner_user_id="user-1")

        assert result is not None
        assert result.name == "Sarah"

    async def test_add_memory_rejects_other_user_person(self, memory_store):
        """Test that memory cannot reference another user's person."""
        person = await memory_store.create_person(owner_user_id="user-1", name="Sarah")

        with pytest.raises(ValueError, match="Invalid subject person ID"):
            await memory_store.add_memory(
                content="Fact about Sarah",
                owner_user_id="user-2",
                subject_person_ids=[person.id],
            )

    async def test_add_memory_accepts_own_person(self, memory_store):
        """Test that memory can reference owner's person."""
        person = await memory_store.create_person(owner_user_id="user-1", name="Sarah")

        memory = await memory_store.add_memory(
            content="Sarah likes coffee",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )

        assert memory.subject_person_ids == [person.id]
