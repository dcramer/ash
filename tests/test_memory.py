"""Tests for memory system behavior.

Tests focus on:
- Supersession logic (core business behavior)
- Garbage collection and eviction
- Scoping rules (personal vs group)
- Error handling

Note: These tests use the new FileMemoryStore architecture.
For lower-level FileMemoryStore tests, see test_memory_file_store.py.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.memory import MemoryManager
from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.file_store import FileMemoryStore
from ash.memory.index import VectorIndex, VectorSearchResult
from ash.memory.types import MemoryType


def _make_search_results(
    results: list[tuple[str, float]],
) -> list[VectorSearchResult]:
    """Convert (memory_id, similarity) tuples to VectorSearchResult objects."""
    return [VectorSearchResult(memory_id=mid, similarity=sim) for mid, sim in results]


@pytest.fixture
def mock_embedding_generator():
    """Create a mock embedding generator."""
    generator = MagicMock(spec=EmbeddingGenerator)
    # Return mock embedding (1536 dimensions)
    generator.generate = AsyncMock(return_value=[0.1] * 1536)
    return generator


@pytest.fixture
def mock_index():
    """Create a mock vector index."""
    index = MagicMock(spec=VectorIndex)
    index.search = AsyncMock(return_value=[])
    index.add_embedding = AsyncMock()
    index.delete_embedding = AsyncMock()
    return index


@pytest.fixture
async def memory_manager(
    file_memory_store, mock_index, mock_embedding_generator, db_session
):
    """Create a memory manager with mocked components."""
    return MemoryManager(
        store=file_memory_store,
        index=mock_index,
        embedding_generator=mock_embedding_generator,
        db_session=db_session,
    )


class TestMemorySupersession:
    """Tests for memory supersession - core business logic."""

    async def test_mark_memory_superseded(self, file_memory_store: FileMemoryStore):
        """Test marking a memory as superseded."""
        old_memory = await file_memory_store.add_memory(
            content="User's favorite color is red",
            owner_user_id="user-1",
        )
        new_memory = await file_memory_store.add_memory(
            content="User's favorite color is blue",
            owner_user_id="user-1",
        )

        result = await file_memory_store.mark_memory_superseded(
            memory_id=old_memory.id,
            superseded_by_id=new_memory.id,
        )

        assert result is True
        old_refreshed = await file_memory_store.get_memory(old_memory.id)
        assert old_refreshed is not None
        assert old_refreshed.superseded_at is not None
        assert old_refreshed.superseded_by_id == new_memory.id

    async def test_get_memories_excludes_superseded_by_default(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that get_memories excludes superseded memories."""
        old_memory = await file_memory_store.add_memory(content="Old fact")
        new_memory = await file_memory_store.add_memory(content="New fact")
        await file_memory_store.mark_memory_superseded(old_memory.id, new_memory.id)

        memories = await file_memory_store.get_memories(include_superseded=False)

        assert len(memories) == 1
        assert memories[0].content == "New fact"

    async def test_get_memories_can_include_superseded(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that get_memories can include superseded memories."""
        old_memory = await file_memory_store.add_memory(content="Old fact")
        new_memory = await file_memory_store.add_memory(content="New fact")
        await file_memory_store.mark_memory_superseded(old_memory.id, new_memory.id)

        memories = await file_memory_store.get_memories(include_superseded=True)

        assert len(memories) == 2


class TestMemoryManagerSupersession:
    """Tests for automatic supersession via MemoryManager."""

    async def test_add_memory_supersedes_conflicting(
        self, memory_manager, file_memory_store, mock_index
    ):
        """Test that adding a memory supersedes high-similarity conflicts."""
        old_memory = await file_memory_store.add_memory(
            content="User's favorite color is red",
            owner_user_id="user-1",
        )

        # Mock the index to return the old memory as similar
        mock_index.search.return_value = _make_search_results(
            [
                (old_memory.id, 0.85)  # Above 0.75 threshold
            ]
        )

        new_memory = await memory_manager.add_memory(
            content="User's favorite color is blue",
            owner_user_id="user-1",
        )

        old_refreshed = await file_memory_store.get_memory(old_memory.id)
        assert old_refreshed.superseded_at is not None
        assert old_refreshed.superseded_by_id == new_memory.id

    async def test_no_supersession_below_threshold(
        self, memory_manager, file_memory_store, mock_index
    ):
        """Test that memories below similarity threshold are not superseded."""
        old_memory = await file_memory_store.add_memory(
            content="User likes pizza",
            owner_user_id="user-1",
        )

        # Mock the index to return low similarity
        mock_index.search.return_value = _make_search_results(
            [
                (old_memory.id, 0.5)  # Below 0.75 threshold
            ]
        )

        await memory_manager.add_memory(
            content="User likes coffee",
            owner_user_id="user-1",
        )

        old_refreshed = await file_memory_store.get_memory(old_memory.id)
        assert old_refreshed.superseded_at is None


class TestGarbageCollection:
    """Tests for memory garbage collection."""

    async def test_gc_removes_expired_memories(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that GC removes expired memories."""
        past = datetime.now(UTC) - timedelta(days=1)
        expired = await file_memory_store.add_memory(content="Expired", expires_at=past)
        valid = await file_memory_store.add_memory(content="Valid")

        result = await file_memory_store.gc()

        assert result.removed_count == 1
        assert expired.id in result.archived_ids
        assert await file_memory_store.get_memory(expired.id) is None
        assert await file_memory_store.get_memory(valid.id) is not None

    async def test_gc_removes_superseded_memories(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that GC removes superseded memories."""
        old = await file_memory_store.add_memory(content="Old fact")
        new = await file_memory_store.add_memory(content="New fact")
        await file_memory_store.mark_memory_superseded(old.id, new.id)

        result = await file_memory_store.gc()

        assert result.removed_count == 1
        assert old.id in result.archived_ids
        assert await file_memory_store.get_memory(old.id) is None
        assert await file_memory_store.get_memory(new.id) is not None

    async def test_gc_handles_empty_state(self, file_memory_store: FileMemoryStore):
        """Test that GC works with only valid memories."""
        await file_memory_store.add_memory(content="Valid 1")
        await file_memory_store.add_memory(content="Valid 2")

        result = await file_memory_store.gc()

        assert result.removed_count == 0


class TestEnforceMaxEntries:
    """Tests for max_entries eviction policy."""

    async def test_evicts_oldest_when_over_limit(
        self, memory_manager, file_memory_store
    ):
        """Test that enforce_max_entries evicts oldest memories."""
        old_time = datetime.now(UTC) - timedelta(days=10)
        memories = []
        for i in range(5):
            m = await file_memory_store.add_memory(content=f"Fact {i}")
            m.created_at = old_time + timedelta(hours=i)
            await file_memory_store.update_memory(m)
            memories.append(m)

        evicted = await memory_manager.enforce_max_entries(3)

        assert evicted == 2
        remaining = await file_memory_store.get_memories()
        assert len(remaining) == 3

    async def test_no_eviction_when_under_limit(
        self, memory_manager, file_memory_store
    ):
        """Test that no eviction happens when under limit."""
        await file_memory_store.add_memory(content="Fact 1")
        await file_memory_store.add_memory(content="Fact 2")

        evicted = await memory_manager.enforce_max_entries(10)

        assert evicted == 0


class TestScoping:
    """Tests for memory scoping (personal vs group)."""

    async def test_personal_scope_filters_by_user(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that personal memories are filtered by user."""
        await file_memory_store.add_memory(
            content="User 1 fact", owner_user_id="user-1"
        )
        await file_memory_store.add_memory(
            content="User 2 fact", owner_user_id="user-2"
        )

        user1_memories = await file_memory_store.get_memories(owner_user_id="user-1")

        assert len(user1_memories) == 1
        assert user1_memories[0].content == "User 1 fact"

    async def test_group_scope_filters_by_chat(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that group memories are filtered by chat."""
        await file_memory_store.add_memory(content="Chat 1 fact", chat_id="chat-1")
        await file_memory_store.add_memory(content="Chat 2 fact", chat_id="chat-2")

        chat1_memories = await file_memory_store.get_memories(chat_id="chat-1")

        assert len(chat1_memories) == 1
        assert chat1_memories[0].content == "Chat 1 fact"


class TestEphemeralDecay:
    """Tests for ephemeral memory type decay."""

    async def test_observation_decays_after_ttl(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that observation memories decay after 3 days."""
        old_time = datetime.now(UTC) - timedelta(days=5)
        observation = await file_memory_store.add_memory(
            content="Noticed something",
            memory_type=MemoryType.OBSERVATION,
        )
        observation.created_at = old_time
        await file_memory_store.update_memory(observation)

        result = await file_memory_store.gc()

        assert result.removed_count == 1
        assert observation.id in result.archived_ids

    async def test_preference_never_decays(self, file_memory_store: FileMemoryStore):
        """Test that preference memories never decay."""
        old_time = datetime.now(UTC) - timedelta(days=365)
        preference = await file_memory_store.add_memory(
            content="Likes dark mode",
            memory_type=MemoryType.PREFERENCE,
        )
        preference.created_at = old_time
        await file_memory_store.update_memory(preference)

        result = await file_memory_store.gc()

        assert result.removed_count == 0
        assert await file_memory_store.get_memory(preference.id) is not None


class TestCrossContextRetrieval:
    """Tests for cross-context memory retrieval."""

    async def test_find_memories_about_user_by_username(
        self, file_memory_store: FileMemoryStore
    ):
        """Test finding memories about a user by username."""
        # Create a person with a username alias
        person = await file_memory_store.create_person(
            owner_user_id="alice",
            name="Bob",
            aliases=["@bob", "bob"],
        )

        # Create a memory about this person
        await file_memory_store.add_memory(
            content="Bob likes pizza",
            owner_user_id="alice",
            subject_person_ids=[person.id],
        )

        # Find memories about bob
        memories = await file_memory_store.find_memories_about_user(username="bob")

        assert len(memories) == 1
        assert memories[0].content == "Bob likes pizza"

    async def test_find_memories_about_user_excludes_owner(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that find_memories_about_user can exclude owner's memories."""
        # Create a person record for both Alice and Carol
        person_alice = await file_memory_store.create_person(
            owner_user_id="alice",
            name="Bob",
            aliases=["bob"],
        )
        person_carol = await file_memory_store.create_person(
            owner_user_id="carol",
            name="Bob",
            aliases=["bob"],
        )

        # Alice has a memory about Bob
        await file_memory_store.add_memory(
            content="Bob likes pizza",
            owner_user_id="alice",
            subject_person_ids=[person_alice.id],
        )

        # Carol also has a memory about Bob
        await file_memory_store.add_memory(
            content="Bob likes pasta",
            owner_user_id="carol",
            subject_person_ids=[person_carol.id],
        )

        # Find memories about bob excluding alice's
        memories = await file_memory_store.find_memories_about_user(
            username="bob", exclude_owner_user_id="alice"
        )

        assert len(memories) == 1
        assert memories[0].content == "Bob likes pasta"
        assert memories[0].owner_user_id == "carol"

    async def test_find_memories_about_user_with_at_prefix(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that @username prefix is handled."""
        person = await file_memory_store.create_person(
            owner_user_id="alice",
            name="Bob",
            aliases=["bob"],
        )

        await file_memory_store.add_memory(
            content="Bob is cool",
            owner_user_id="alice",
            subject_person_ids=[person.id],
        )

        # Find with @ prefix
        memories = await file_memory_store.find_memories_about_user(username="@bob")

        assert len(memories) == 1


class TestSecretsFiltering:
    """Tests for secrets filtering in MemoryManager."""

    async def test_rejects_openai_api_key(self, memory_manager):
        """Test that OpenAI API keys are rejected."""
        with pytest.raises(ValueError, match="potential secrets"):
            await memory_manager.add_memory(
                content="My API key is sk-abc123def456ghi789abcdef",
                owner_user_id="user-1",
            )

    async def test_rejects_github_token(self, memory_manager):
        """Test that GitHub tokens are rejected."""
        with pytest.raises(ValueError, match="potential secrets"):
            await memory_manager.add_memory(
                content="Use token: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                owner_user_id="user-1",
            )

    async def test_rejects_aws_access_key(self, memory_manager):
        """Test that AWS access keys are rejected."""
        with pytest.raises(ValueError, match="potential secrets"):
            await memory_manager.add_memory(
                content="AWS key: AKIAIOSFODNN7EXAMPLE",
                owner_user_id="user-1",
            )

    async def test_rejects_credit_card(self, memory_manager):
        """Test that credit card numbers are rejected."""
        with pytest.raises(ValueError, match="potential secrets"):
            await memory_manager.add_memory(
                content="Card: 4111-1111-1111-1111",
                owner_user_id="user-1",
            )

    async def test_rejects_ssn(self, memory_manager):
        """Test that Social Security Numbers are rejected."""
        with pytest.raises(ValueError, match="potential secrets"):
            await memory_manager.add_memory(
                content="SSN: 123-45-6789",
                owner_user_id="user-1",
            )

    async def test_rejects_password_is_pattern(self, memory_manager):
        """Test that 'password is X' patterns are rejected."""
        with pytest.raises(ValueError, match="potential secrets"):
            await memory_manager.add_memory(
                content="Remember my password is hunter2",
                owner_user_id="user-1",
            )

    async def test_rejects_password_colon_pattern(self, memory_manager):
        """Test that 'password: X' patterns are rejected."""
        with pytest.raises(ValueError, match="potential secrets"):
            await memory_manager.add_memory(
                content="My password: secret123",
                owner_user_id="user-1",
            )

    async def test_rejects_private_key(self, memory_manager):
        """Test that private keys are rejected."""
        with pytest.raises(ValueError, match="potential secrets"):
            await memory_manager.add_memory(
                content="-----BEGIN PRIVATE KEY-----\nMIIEvg...",
                owner_user_id="user-1",
            )

    async def test_allows_safe_content(self, memory_manager):
        """Test that normal content is allowed."""
        memory = await memory_manager.add_memory(
            content="User prefers dark mode",
            owner_user_id="user-1",
        )
        assert memory.content == "User prefers dark mode"

    async def test_allows_number_sequences(self, memory_manager):
        """Test that normal number sequences are allowed."""
        # 4 digits not credit card, no SSN pattern
        memory = await memory_manager.add_memory(
            content="Favorite number is 1234",
            owner_user_id="user-1",
        )
        assert memory.content == "Favorite number is 1234"


class TestSensitivity:
    """Tests for memory sensitivity classification."""

    async def test_add_memory_with_sensitivity(
        self, file_memory_store: FileMemoryStore
    ):
        """Test adding a memory with sensitivity classification."""
        from ash.memory.types import Sensitivity

        memory = await file_memory_store.add_memory(
            content="Has anxiety",
            owner_user_id="user-1",
            sensitivity=Sensitivity.SENSITIVE,
        )

        assert memory.sensitivity == Sensitivity.SENSITIVE

        # Verify it persists
        loaded = await file_memory_store.get_memory(memory.id)
        assert loaded is not None
        assert loaded.sensitivity == Sensitivity.SENSITIVE

    async def test_sensitivity_none_means_public(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that None sensitivity is treated as public."""
        memory = await file_memory_store.add_memory(
            content="Likes pizza",
            owner_user_id="user-1",
        )

        # No sensitivity set
        assert memory.sensitivity is None

    async def test_memory_serialization_with_sensitivity(
        self, file_memory_store: FileMemoryStore
    ):
        """Test that sensitivity is correctly serialized/deserialized."""
        from ash.memory.types import Sensitivity

        memory = await file_memory_store.add_memory(
            content="Personal info",
            owner_user_id="user-1",
            sensitivity=Sensitivity.PERSONAL,
        )

        # Force cache invalidation to ensure we read from disk
        file_memory_store._invalidate_memories_cache()

        loaded = await file_memory_store.get_memory(memory.id)
        assert loaded is not None
        assert loaded.sensitivity == Sensitivity.PERSONAL
