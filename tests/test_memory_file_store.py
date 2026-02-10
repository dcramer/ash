"""Tests for filesystem-based memory storage.

Tests focus on:
- FileMemoryStore CRUD operations
- JSONL serialization/deserialization
- GC with ephemeral decay
- Supersession chain tracking
- Migration from SQLite
"""

from datetime import UTC, datetime, timedelta

from ash.memory.file_store import FileMemoryStore
from ash.memory.types import MemoryEntry, MemoryType


class TestFileMemoryStore:
    """Tests for FileMemoryStore basic operations."""

    async def test_add_memory(self, file_memory_store: FileMemoryStore):
        """Test adding a memory entry."""
        memory = await file_memory_store.add_memory(
            content="User prefers dark mode",
            memory_type=MemoryType.PREFERENCE,
            source="user",
            owner_user_id="user-1",
        )

        assert memory.id is not None
        assert memory.content == "User prefers dark mode"
        assert memory.memory_type == MemoryType.PREFERENCE
        assert memory.source == "user"
        assert memory.owner_user_id == "user-1"

    async def test_get_memory(self, file_memory_store: FileMemoryStore):
        """Test getting a memory by ID."""
        memory = await file_memory_store.add_memory(
            content="Test fact",
            owner_user_id="user-1",
        )

        retrieved = await file_memory_store.get_memory(memory.id)

        assert retrieved is not None
        assert retrieved.id == memory.id
        assert retrieved.content == memory.content

    async def test_get_memory_by_prefix(self, file_memory_store: FileMemoryStore):
        """Test getting a memory by ID prefix."""
        memory = await file_memory_store.add_memory(
            content="Test fact",
            owner_user_id="user-1",
        )

        # Use first 8 characters as prefix
        prefix = memory.id[:8]
        retrieved = await file_memory_store.get_memory_by_prefix(prefix)

        assert retrieved is not None
        assert retrieved.id == memory.id

    async def test_get_memories_with_filters(self, file_memory_store: FileMemoryStore):
        """Test getting memories with various filters."""
        await file_memory_store.add_memory(
            content="User 1 fact",
            owner_user_id="user-1",
        )
        await file_memory_store.add_memory(
            content="User 2 fact",
            owner_user_id="user-2",
        )

        memories = await file_memory_store.get_memories(owner_user_id="user-1")

        assert len(memories) == 1
        assert memories[0].content == "User 1 fact"

    async def test_delete_memory(self, file_memory_store: FileMemoryStore):
        """Test deleting a memory."""
        memory = await file_memory_store.add_memory(
            content="To be deleted",
            owner_user_id="user-1",
        )

        result = await file_memory_store.delete_memory(
            memory.id, owner_user_id="user-1"
        )

        assert result is True
        assert await file_memory_store.get_memory(memory.id) is None

    async def test_delete_memory_authorization(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that authorization prevents unauthorized deletion."""
        memory = await file_memory_store.add_memory(
            content="Secret",
            owner_user_id="user-1",
        )

        result = await file_memory_store.delete_memory(
            memory.id, owner_user_id="user-2"
        )

        assert result is False
        assert await file_memory_store.get_memory(memory.id) is not None


class TestMemorySupersession:
    """Tests for memory supersession."""

    async def test_mark_memory_superseded(self, file_memory_store: FileMemoryStore):
        """Test marking a memory as superseded."""
        old_memory = await file_memory_store.add_memory(
            content="Favorite color is red",
            owner_user_id="user-1",
        )
        new_memory = await file_memory_store.add_memory(
            content="Favorite color is blue",
            owner_user_id="user-1",
        )

        result = await file_memory_store.mark_memory_superseded(
            old_memory.id, new_memory.id
        )

        assert result is True
        old_refreshed = await file_memory_store.get_memory(old_memory.id)
        assert old_refreshed.superseded_at is not None
        assert old_refreshed.superseded_by_id == new_memory.id

    async def test_get_memories_excludes_superseded(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that superseded memories are excluded by default."""
        old_memory = await file_memory_store.add_memory(content="Old fact")
        new_memory = await file_memory_store.add_memory(content="New fact")
        await file_memory_store.mark_memory_superseded(old_memory.id, new_memory.id)

        memories = await file_memory_store.get_memories()

        assert len(memories) == 1
        assert memories[0].content == "New fact"

    async def test_get_memories_can_include_superseded(
        self, file_memory_store: FileMemoryStore
    ):
        """Test including superseded memories."""
        old_memory = await file_memory_store.add_memory(content="Old fact")
        new_memory = await file_memory_store.add_memory(content="New fact")
        await file_memory_store.mark_memory_superseded(old_memory.id, new_memory.id)

        memories = await file_memory_store.get_memories(include_superseded=True)

        assert len(memories) == 2


class TestGarbageCollection:
    """Tests for garbage collection."""

    async def test_gc_archives_expired_memories(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that GC archives expired memories."""
        past = datetime.now(UTC) - timedelta(days=1)
        expired = await file_memory_store.add_memory(content="Expired", expires_at=past)
        valid = await file_memory_store.add_memory(content="Valid")

        result = await file_memory_store.gc()

        assert result.removed_count == 1
        assert expired.id in result.archived_ids
        assert await file_memory_store.get_memory(expired.id) is None
        assert await file_memory_store.get_memory(valid.id) is not None

    async def test_gc_archives_superseded_memories(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that GC archives superseded memories."""
        old = await file_memory_store.add_memory(content="Old fact")
        new = await file_memory_store.add_memory(content="New fact")
        await file_memory_store.mark_memory_superseded(old.id, new.id)

        result = await file_memory_store.gc()

        assert result.removed_count == 1
        assert old.id in result.archived_ids
        assert await file_memory_store.get_memory(old.id) is None
        assert await file_memory_store.get_memory(new.id) is not None

    async def test_gc_ephemeral_decay(self, file_memory_store: FileMemoryStore):
        """Test that ephemeral memories decay based on type TTL."""
        # Create an old observation (TTL: 3 days)
        old_observation = await file_memory_store.add_memory(
            content="Old observation",
            memory_type=MemoryType.OBSERVATION,
        )
        # Manually set created_at to 5 days ago
        old_observation.created_at = datetime.now(UTC) - timedelta(days=5)
        await file_memory_store.update_memory(old_observation)

        # Create a new context memory (TTL: 7 days) - 2 days old
        new_context = await file_memory_store.add_memory(
            content="New context",
            memory_type=MemoryType.CONTEXT,
        )

        result = await file_memory_store.gc()

        # Old observation should be archived (5 days > 3 day TTL)
        # New context should remain (2 days < 7 day TTL)
        assert result.removed_count == 1
        assert old_observation.id in result.archived_ids
        assert await file_memory_store.get_memory(new_context.id) is not None

    async def test_gc_preserves_long_lived_types(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that long-lived memory types don't decay."""
        # Create an old preference (no TTL)
        old_preference = await file_memory_store.add_memory(
            content="Old preference",
            memory_type=MemoryType.PREFERENCE,
        )
        old_preference.created_at = datetime.now(UTC) - timedelta(days=100)
        await file_memory_store.update_memory(old_preference)

        result = await file_memory_store.gc()

        assert result.removed_count == 0
        assert await file_memory_store.get_memory(old_preference.id) is not None

    async def test_gc_writes_to_archive(self, file_memory_store: FileMemoryStore):
        """Test that GC writes to archive file."""
        past = datetime.now(UTC) - timedelta(days=1)
        expired = await file_memory_store.add_memory(
            content="To be archived", expires_at=past
        )

        await file_memory_store.gc()

        archived = await file_memory_store.get_archived_memories()
        assert len(archived) == 1
        assert archived[0].id == expired.id
        assert archived[0].archive_reason == "expired"
        assert archived[0].archived_at is not None


class TestSupersessionChain:
    """Tests for supersession chain tracking."""

    async def test_get_supersession_chain(self, file_memory_store: FileMemoryStore):
        """Test getting the chain of superseded memories."""
        # Create a chain: v1 -> v2 -> v3
        v1 = await file_memory_store.add_memory(content="Color is red")
        v2 = await file_memory_store.add_memory(content="Color is green")
        v3 = await file_memory_store.add_memory(content="Color is blue")

        await file_memory_store.mark_memory_superseded(v1.id, v2.id)
        await file_memory_store.mark_memory_superseded(v2.id, v3.id)

        # Archive the superseded entries
        await file_memory_store.gc()

        # Get chain leading to v3
        chain = await file_memory_store.get_supersession_chain(v3.id)

        # Should have v1 and v2 (oldest first)
        assert len(chain) == 2
        assert chain[0].content == "Color is red"
        assert chain[1].content == "Color is green"


class TestPersonOperations:
    """Tests for person entity operations."""

    async def test_create_person(self, file_memory_store: FileMemoryStore):
        """Test creating a person."""
        person = await file_memory_store.create_person(
            owner_user_id="user-1",
            name="Sarah",
            relationship="wife",
            aliases=["my wife"],
        )

        assert person.id is not None
        assert person.name == "Sarah"
        assert person.relationship == "wife"
        assert person.aliases == ["my wife"]

    async def test_find_person_by_name(self, file_memory_store: FileMemoryStore):
        """Test finding a person by name."""
        await file_memory_store.create_person(
            owner_user_id="user-1",
            name="Sarah",
        )

        found = await file_memory_store.find_person_by_reference("user-1", "Sarah")

        assert found is not None
        assert found.name == "Sarah"

    async def test_find_person_by_relationship(
        self, file_memory_store: FileMemoryStore
    ):
        """Test finding a person by relationship."""
        await file_memory_store.create_person(
            owner_user_id="user-1",
            name="Sarah",
            relationship="wife",
        )

        found = await file_memory_store.find_person_by_reference("user-1", "wife")

        assert found is not None
        assert found.name == "Sarah"

    async def test_find_person_by_alias(self, file_memory_store: FileMemoryStore):
        """Test finding a person by alias."""
        await file_memory_store.create_person(
            owner_user_id="user-1",
            name="Sarah",
            aliases=["my wife"],
        )

        found = await file_memory_store.find_person_by_reference("user-1", "my wife")

        assert found is not None
        assert found.name == "Sarah"

    async def test_add_person_alias(self, file_memory_store: FileMemoryStore):
        """Test adding an alias to a person."""
        person = await file_memory_store.create_person(
            owner_user_id="user-1",
            name="Sarah",
        )

        updated = await file_memory_store.add_person_alias(person.id, "honey", "user-1")

        assert updated is not None
        assert "honey" in updated.aliases


class TestMemoryTypes:
    """Tests for memory type handling."""

    async def test_memory_type_serialization(self, file_memory_store: FileMemoryStore):
        """Test that memory types are correctly serialized/deserialized."""
        for memory_type in MemoryType:
            memory = await file_memory_store.add_memory(
                content=f"Test {memory_type.value}",
                memory_type=memory_type,
            )

            # Invalidate cache to force reload
            file_memory_store._invalidate_memories_cache()

            retrieved = await file_memory_store.get_memory(memory.id)
            assert retrieved.memory_type == memory_type

    async def test_memory_entry_to_dict(self):
        """Test MemoryEntry serialization."""
        now = datetime.now(UTC)
        entry = MemoryEntry(
            id="test-id",
            version=1,
            content="Test content",
            memory_type=MemoryType.PREFERENCE,
            embedding="base64data",
            created_at=now,
            owner_user_id="user-1",
            subject_person_ids=["person-1"],
            source="user",
        )

        d = entry.to_dict()

        assert d["id"] == "test-id"
        assert d["memory_type"] == "preference"
        assert d["subject_person_ids"] == ["person-1"]

    async def test_memory_entry_from_dict(self):
        """Test MemoryEntry deserialization."""
        d = {
            "id": "test-id",
            "version": 1,
            "content": "Test content",
            "memory_type": "preference",
            "embedding": "base64data",
            "created_at": "2026-02-09T10:00:00+00:00",
            "owner_user_id": "user-1",
            "source": "user",
        }

        entry = MemoryEntry.from_dict(d)

        assert entry.id == "test-id"
        assert entry.memory_type == MemoryType.PREFERENCE
        assert entry.owner_user_id == "user-1"


class TestCacheInvalidation:
    """Tests for cache invalidation on file changes."""

    async def test_cache_invalidates_after_write(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that cache is invalidated after writes."""
        await file_memory_store.add_memory(content="First")
        memories1 = await file_memory_store.get_memories()
        assert len(memories1) == 1

        await file_memory_store.add_memory(content="Second")
        memories2 = await file_memory_store.get_memories()
        assert len(memories2) == 2
