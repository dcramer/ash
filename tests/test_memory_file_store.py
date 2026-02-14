"""Tests for filesystem-based memory storage.

Tests focus on:
- FileMemoryStore CRUD operations
- JSONL serialization/deserialization
- GC with ephemeral decay
- Supersession chain tracking
- Compaction
- Embedding storage
- Archive behavior
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
        assert old_refreshed is not None
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
            assert retrieved is not None
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


class TestPortableMemories:
    """Tests for portable field on cross-context retrieval."""

    async def test_non_portable_excluded_from_find_by_subject(
        self, file_memory_store: FileMemoryStore
    ):
        """Non-portable memories are excluded from find_memories_by_subject."""
        person_id = "person-bob"

        # Portable memory
        await file_memory_store.add_memory(
            content="Bob loves pizza",
            owner_user_id="alice",
            subject_person_ids=[person_id],
            portable=True,
        )

        # Non-portable memory
        await file_memory_store.add_memory(
            content="Bob is presenting next",
            owner_user_id="alice",
            subject_person_ids=[person_id],
            portable=False,
        )

        memories = await file_memory_store.find_memories_by_subject(
            person_ids={person_id}
        )

        assert len(memories) == 1
        assert memories[0].content == "Bob loves pizza"

    async def test_portable_default_is_true(self, file_memory_store: FileMemoryStore):
        """Memories default to portable=True."""
        memory = await file_memory_store.add_memory(
            content="Some fact",
            owner_user_id="user-1",
        )
        assert memory.portable is True

    async def test_non_portable_serialization(self, file_memory_store: FileMemoryStore):
        """Non-portable flag survives serialization round-trip."""
        memory = await file_memory_store.add_memory(
            content="Ephemeral fact",
            owner_user_id="user-1",
            subject_person_ids=["person-1"],
            portable=False,
        )

        # Force cache reload
        file_memory_store._invalidate_memories_cache()

        loaded = await file_memory_store.get_memory(memory.id)
        assert loaded is not None
        assert loaded.portable is False


class TestRemapSubjectPersonId:
    """Tests for remap_subject_person_id after merge."""

    async def test_remap_updates_matching_memories(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that old person ID is replaced with new in subject_person_ids."""
        m1 = await file_memory_store.add_memory(
            content="Sarah likes hiking",
            subject_person_ids=["old-id"],
        )
        m2 = await file_memory_store.add_memory(
            content="Bob likes coding",
            subject_person_ids=["other-id"],
        )

        count = await file_memory_store.remap_subject_person_id("old-id", "new-id")

        assert count == 1
        updated = await file_memory_store.get_memory(m1.id)
        assert updated is not None
        assert updated.subject_person_ids == ["new-id"]

        # Unrelated memory should not be touched
        untouched = await file_memory_store.get_memory(m2.id)
        assert untouched is not None
        assert untouched.subject_person_ids == ["other-id"]

    async def test_remap_handles_multiple_subjects(
        self, file_memory_store: FileMemoryStore
    ):
        """Test remap when memory has multiple subject_person_ids."""
        m = await file_memory_store.add_memory(
            content="Sarah and Bob went hiking",
            subject_person_ids=["old-id", "bob-id"],
        )

        count = await file_memory_store.remap_subject_person_id("old-id", "new-id")

        assert count == 1
        updated = await file_memory_store.get_memory(m.id)
        assert updated is not None
        assert updated.subject_person_ids == ["new-id", "bob-id"]

    async def test_remap_returns_zero_when_no_matches(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that remap returns 0 when no memories match."""
        await file_memory_store.add_memory(
            content="Unrelated",
            subject_person_ids=["other-id"],
        )

        count = await file_memory_store.remap_subject_person_id("old-id", "new-id")

        assert count == 0

    async def test_remap_no_memories(self, file_memory_store: FileMemoryStore):
        """Test remap on empty store."""
        count = await file_memory_store.remap_subject_person_id("old-id", "new-id")
        assert count == 0


class TestDeleteMemoryArchives:
    """Tests that delete_memory archives instead of physically removing."""

    async def test_delete_memory_archives_in_place(
        self, file_memory_store: FileMemoryStore
    ):
        """Deleted memory should be archived, not removed from file."""
        memory = await file_memory_store.add_memory(
            content="Delete me",
            owner_user_id="user-1",
        )

        await file_memory_store.delete_memory(memory.id, owner_user_id="user-1")

        # Should not appear in active queries
        assert await file_memory_store.get_memory(memory.id) is None

        # Should appear in archived
        archived = await file_memory_store.get_archived_memories()
        assert len(archived) == 1
        assert archived[0].id == memory.id
        assert archived[0].archive_reason == "user_deleted"
        assert archived[0].archived_at is not None

    async def test_delete_memory_still_in_get_all(
        self, file_memory_store: FileMemoryStore
    ):
        """Deleted memory should appear in get_all_memories."""
        memory = await file_memory_store.add_memory(
            content="Delete me",
            owner_user_id="user-1",
        )

        await file_memory_store.delete_memory(memory.id, owner_user_id="user-1")

        all_memories = await file_memory_store.get_all_memories()
        assert len(all_memories) == 1
        assert all_memories[0].id == memory.id


class TestGetMemoriesExcludesArchived:
    """Tests that all retrieval methods exclude archived memories."""

    async def test_get_memories_excludes_archived(
        self, file_memory_store: FileMemoryStore
    ):
        """get_memories should not return archived entries."""
        m1 = await file_memory_store.add_memory(content="Active", owner_user_id="u1")
        m2 = await file_memory_store.add_memory(content="Archived", owner_user_id="u1")

        await file_memory_store.archive_memories({m2.id}, "test")

        memories = await file_memory_store.get_memories(owner_user_id="u1")
        assert len(memories) == 1
        assert memories[0].id == m1.id

    async def test_get_memory_excludes_archived(
        self, file_memory_store: FileMemoryStore
    ):
        """get_memory should return None for archived entries."""
        memory = await file_memory_store.add_memory(content="Will archive")
        await file_memory_store.archive_memories({memory.id}, "test")

        assert await file_memory_store.get_memory(memory.id) is None

    async def test_get_memory_by_prefix_excludes_archived(
        self, file_memory_store: FileMemoryStore
    ):
        """get_memory_by_prefix should not match archived entries."""
        memory = await file_memory_store.add_memory(content="Will archive")
        prefix = memory.id[:8]
        await file_memory_store.archive_memories({memory.id}, "test")

        assert await file_memory_store.get_memory_by_prefix(prefix) is None

    async def test_find_memories_by_subject_excludes_archived(
        self, file_memory_store: FileMemoryStore
    ):
        """find_memories_by_subject should skip archived entries."""
        pid = "person-1"
        await file_memory_store.add_memory(
            content="Active about person",
            subject_person_ids=[pid],
            owner_user_id="u1",
        )
        m2 = await file_memory_store.add_memory(
            content="Archived about person",
            subject_person_ids=[pid],
            owner_user_id="u1",
        )
        await file_memory_store.archive_memories({m2.id}, "test")

        results = await file_memory_store.find_memories_by_subject(person_ids={pid})
        assert len(results) == 1
        assert results[0].content == "Active about person"


class TestCompact:
    """Tests for compact() — permanent removal of old archived entries."""

    async def test_compact_removes_old_archived(
        self, file_memory_store: FileMemoryStore
    ):
        """Compact should permanently remove entries archived > N days ago."""
        active = await file_memory_store.add_memory(content="Active")
        to_archive = await file_memory_store.add_memory(content="Old archived")

        # Archive it
        await file_memory_store.archive_memories({to_archive.id}, "test")

        # Backdate the archived_at to 100 days ago
        memories = await file_memory_store.get_all_memories()
        for m in memories:
            if m.id == to_archive.id:
                m.archived_at = datetime.now(UTC) - timedelta(days=100)
        await file_memory_store._memories_jsonl.rewrite(memories)
        file_memory_store._invalidate_memories_cache()

        removed = await file_memory_store.compact(older_than_days=90)

        assert removed == 1
        all_remaining = await file_memory_store.get_all_memories()
        assert len(all_remaining) == 1
        assert all_remaining[0].id == active.id

    async def test_compact_preserves_recent_archived(
        self, file_memory_store: FileMemoryStore
    ):
        """Compact should keep entries archived less than N days ago."""
        to_archive = await file_memory_store.add_memory(content="Recently archived")
        await file_memory_store.archive_memories({to_archive.id}, "test")

        removed = await file_memory_store.compact(older_than_days=90)

        assert removed == 0
        all_remaining = await file_memory_store.get_all_memories()
        assert len(all_remaining) == 1

    async def test_compact_preserves_active(self, file_memory_store: FileMemoryStore):
        """Compact should never touch active (non-archived) entries."""
        await file_memory_store.add_memory(content="Active 1")
        await file_memory_store.add_memory(content="Active 2")

        removed = await file_memory_store.compact(older_than_days=0)

        assert removed == 0
        all_remaining = await file_memory_store.get_all_memories()
        assert len(all_remaining) == 2

    async def test_compact_cleans_orphaned_embeddings(
        self, file_memory_store: FileMemoryStore
    ):
        """Compact should remove embeddings for compacted memories."""
        m1 = await file_memory_store.add_memory(content="Will be compacted")
        m2 = await file_memory_store.add_memory(content="Will stay")

        # Save embeddings for both
        await file_memory_store.save_embedding(m1.id, "emb1")
        await file_memory_store.save_embedding(m2.id, "emb2")

        # Archive m1 and backdate
        await file_memory_store.archive_memories({m1.id}, "test")
        memories = await file_memory_store.get_all_memories()
        for m in memories:
            if m.id == m1.id:
                m.archived_at = datetime.now(UTC) - timedelta(days=100)
        await file_memory_store._memories_jsonl.rewrite(memories)
        file_memory_store._invalidate_memories_cache()

        await file_memory_store.compact(older_than_days=90)

        # Only m2's embedding should remain
        embeddings = await file_memory_store.load_embeddings()
        assert m1.id not in embeddings
        assert m2.id in embeddings
        assert embeddings[m2.id] == "emb2"


class TestEmbeddingStorage:
    """Tests for save_embedding, load_embeddings, get_embedding_for_memory."""

    async def test_save_and_load_embedding(self, file_memory_store: FileMemoryStore):
        """Test basic save and load round-trip."""
        await file_memory_store.save_embedding("mem-1", "base64data1")
        await file_memory_store.save_embedding("mem-2", "base64data2")

        embeddings = await file_memory_store.load_embeddings()

        assert len(embeddings) == 2
        assert embeddings["mem-1"] == "base64data1"
        assert embeddings["mem-2"] == "base64data2"

    async def test_load_embeddings_empty(self, file_memory_store: FileMemoryStore):
        """Test loading when no embeddings exist."""
        embeddings = await file_memory_store.load_embeddings()
        assert embeddings == {}

    async def test_get_embedding_for_memory(self, file_memory_store: FileMemoryStore):
        """Test getting a single embedding."""
        await file_memory_store.save_embedding("mem-1", "base64data")

        result = await file_memory_store.get_embedding_for_memory("mem-1")
        assert result == "base64data"

    async def test_get_embedding_for_memory_missing(
        self, file_memory_store: FileMemoryStore
    ):
        """Test getting a nonexistent embedding returns None."""
        result = await file_memory_store.get_embedding_for_memory("no-such-id")
        assert result is None

    async def test_last_write_wins_for_duplicates(
        self, file_memory_store: FileMemoryStore
    ):
        """When same memory_id is appended twice, last write wins."""
        await file_memory_store.save_embedding("mem-1", "old")
        await file_memory_store.save_embedding("mem-1", "new")

        embeddings = await file_memory_store.load_embeddings()
        assert embeddings["mem-1"] == "new"


class TestFindHearsayBySubject:
    """Tests for find_hearsay_by_subject."""

    async def test_finds_hearsay_about_person(self, file_memory_store: FileMemoryStore):
        """Should find memories about a person spoken by someone else."""
        pid = "person-bob"

        # Hearsay: alice talking about bob
        await file_memory_store.add_memory(
            content="Bob likes hiking",
            owner_user_id="alice",
            subject_person_ids=[pid],
            source_username="alice",
        )

        results = await file_memory_store.find_hearsay_by_subject(
            person_ids={pid},
            source_username="bob",
            owner_user_id="alice",
        )

        assert len(results) == 1
        assert results[0].content == "Bob likes hiking"

    async def test_excludes_self_facts(self, file_memory_store: FileMemoryStore):
        """Should exclude memories where the subject spoke about themselves."""
        pid = "person-bob"

        # Fact: bob talking about himself
        await file_memory_store.add_memory(
            content="I like hiking",
            owner_user_id="alice",
            subject_person_ids=[pid],
            source_username="bob",
        )

        results = await file_memory_store.find_hearsay_by_subject(
            person_ids={pid},
            source_username="bob",
            owner_user_id="alice",
        )

        assert len(results) == 0

    async def test_case_insensitive_username(self, file_memory_store: FileMemoryStore):
        """Username matching should be case-insensitive."""
        pid = "person-bob"

        await file_memory_store.add_memory(
            content="I like coding",
            owner_user_id="alice",
            subject_person_ids=[pid],
            source_username="Bob",  # Capital B
        )

        # Should still be excluded when source_username is "bob" (lowercase)
        results = await file_memory_store.find_hearsay_by_subject(
            person_ids={pid},
            source_username="bob",
            owner_user_id="alice",
        )

        assert len(results) == 0

    async def test_respects_limit(self, file_memory_store: FileMemoryStore):
        """Should stop after reaching the limit."""
        pid = "person-bob"

        for i in range(5):
            await file_memory_store.add_memory(
                content=f"Hearsay {i}",
                owner_user_id="alice",
                subject_person_ids=[pid],
                source_username="charlie",
            )

        results = await file_memory_store.find_hearsay_by_subject(
            person_ids={pid},
            source_username="bob",
            owner_user_id="alice",
            limit=3,
        )

        assert len(results) == 3

    async def test_empty_person_ids_returns_empty(
        self, file_memory_store: FileMemoryStore
    ):
        """Should return empty list for empty person_ids."""
        results = await file_memory_store.find_hearsay_by_subject(
            person_ids=set(),
            source_username="bob",
        )

        assert results == []


class TestClear:
    """Tests for clear() — physical wipe of all data."""

    async def test_clear_removes_all_entries(self, file_memory_store: FileMemoryStore):
        """Clear should physically remove all entries."""
        await file_memory_store.add_memory(content="Active 1")
        await file_memory_store.add_memory(content="Active 2")

        count = await file_memory_store.clear()

        assert count == 2
        all_remaining = await file_memory_store.get_all_memories()
        assert len(all_remaining) == 0

    async def test_clear_removes_archived_entries(
        self, file_memory_store: FileMemoryStore
    ):
        """Clear should also remove archived entries."""
        m1 = await file_memory_store.add_memory(content="Will archive")
        await file_memory_store.add_memory(content="Active")
        await file_memory_store.archive_memories({m1.id}, "test")

        count = await file_memory_store.clear()

        assert count == 2  # Both active and archived
        all_remaining = await file_memory_store.get_all_memories()
        assert len(all_remaining) == 0

    async def test_clear_removes_embeddings(self, file_memory_store: FileMemoryStore):
        """Clear should also wipe embeddings.jsonl."""
        await file_memory_store.add_memory(content="Test")
        await file_memory_store.save_embedding("mem-1", "base64data")

        await file_memory_store.clear()

        embeddings = await file_memory_store.load_embeddings()
        assert embeddings == {}

    async def test_clear_empty_store(self, file_memory_store: FileMemoryStore):
        """Clear on empty store should return 0."""
        count = await file_memory_store.clear()
        assert count == 0


class TestGetAllMemories:
    """Tests for get_all_memories (includes archived/superseded/expired)."""

    async def test_includes_archived(self, file_memory_store: FileMemoryStore):
        """get_all_memories should include archived entries."""
        m1 = await file_memory_store.add_memory(content="Active")
        m2 = await file_memory_store.add_memory(content="To archive")
        await file_memory_store.archive_memories({m2.id}, "test")

        all_memories = await file_memory_store.get_all_memories()
        assert len(all_memories) == 2
        ids = {m.id for m in all_memories}
        assert m1.id in ids
        assert m2.id in ids

    async def test_includes_superseded(self, file_memory_store: FileMemoryStore):
        """get_all_memories should include superseded entries."""
        old = await file_memory_store.add_memory(content="Old")
        new = await file_memory_store.add_memory(content="New")
        await file_memory_store.mark_memory_superseded(old.id, new.id)

        all_memories = await file_memory_store.get_all_memories()
        assert len(all_memories) == 2
