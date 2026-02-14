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

from ash.graph.store import GraphStore
from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.file_store import FileMemoryStore
from ash.memory.index import VectorIndex, VectorSearchResult
from ash.memory.types import MemoryType, Sensitivity


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
    generator.embed = AsyncMock(return_value=[0.1] * 1536)
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
    ash_home, file_memory_store, mock_index, mock_embedding_generator
):
    """Create a GraphStore (replaces MemoryManager) with mocked components."""
    graph_dir = ash_home / "graph"
    graph_dir.mkdir(exist_ok=True)
    return GraphStore(
        memory_store=file_memory_store,
        vector_index=mock_index,
        embedding_generator=mock_embedding_generator,
        people_path=graph_dir / "people.jsonl",
        users_path=graph_dir / "users.jsonl",
        chats_path=graph_dir / "chats.jsonl",
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


class TestGraphStoreSupersession:
    """Tests for automatic supersession via GraphStore."""

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
    """Tests for cross-context memory retrieval using person IDs."""

    async def test_find_memories_by_subject(self, memory_manager, file_memory_store):
        """Test finding memories by subject person IDs."""
        person_id = "person-bob-1"

        # Create a memory about this person
        await file_memory_store.add_memory(
            content="Bob likes pizza",
            owner_user_id="alice",
            subject_person_ids=[person_id],
        )

        # Also a memory not about Bob
        await file_memory_store.add_memory(
            content="General fact",
            owner_user_id="alice",
        )

        # Find memories about bob by person_id via GraphStore
        memories = await memory_manager._find_memories_about_persons(
            person_ids={person_id}
        )

        assert len(memories) == 1
        assert memories[0].content == "Bob likes pizza"

    async def test_find_memories_by_subject_excludes_owner(
        self, memory_manager, file_memory_store
    ):
        """Test that _find_memories_about_persons can exclude owner's memories."""
        person_id_alice = "person-bob-alice"
        person_id_carol = "person-bob-carol"

        # Alice has a memory about Bob
        await file_memory_store.add_memory(
            content="Bob likes pizza",
            owner_user_id="alice",
            subject_person_ids=[person_id_alice],
        )

        # Carol also has a memory about Bob
        await file_memory_store.add_memory(
            content="Bob likes pasta",
            owner_user_id="carol",
            subject_person_ids=[person_id_carol],
        )

        # Find memories about bob excluding alice's
        memories = await memory_manager._find_memories_about_persons(
            person_ids={person_id_alice, person_id_carol},
            exclude_owner_user_id="alice",
        )

        assert len(memories) == 1
        assert memories[0].content == "Bob likes pasta"
        assert memories[0].owner_user_id == "carol"

    async def test_find_memories_by_subject_multiple_ids(
        self, memory_manager, file_memory_store
    ):
        """Test that multiple person IDs find all relevant memories."""
        person_id_1 = "person-bob-1"
        person_id_2 = "person-bob-2"

        await file_memory_store.add_memory(
            content="Bob fact from user1",
            owner_user_id="user1",
            subject_person_ids=[person_id_1],
        )

        await file_memory_store.add_memory(
            content="Bob fact from user2",
            owner_user_id="user2",
            subject_person_ids=[person_id_2],
        )

        memories = await memory_manager._find_memories_about_persons(
            person_ids={person_id_1, person_id_2}
        )

        assert len(memories) == 2


class TestSecretsFiltering:
    """Tests for secrets filtering in GraphStore."""

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


class TestPrivacyFilter:
    """Regression tests for privacy filter bugs.

    Bug 1: _is_user_subject_of_memory always returned False (now fixed by
    accepting pre-resolved person IDs instead of doing sync lookups).

    Bug 2: Inconsistent filter logic between MemoryEntry and SearchResult
    paths (now unified into single _passes_privacy_filter method).
    """

    def test_public_always_passes(self, memory_manager):
        """PUBLIC memories pass regardless of context."""
        assert memory_manager._passes_privacy_filter(
            sensitivity=Sensitivity.PUBLIC,
            subject_person_ids=["person-1"],
            chat_type="group",
            querying_person_ids=set(),
        )

    def test_none_sensitivity_treated_as_public(self, memory_manager):
        """None sensitivity (legacy) treated as PUBLIC."""
        assert memory_manager._passes_privacy_filter(
            sensitivity=None,
            subject_person_ids=["person-1"],
            chat_type="group",
            querying_person_ids=set(),
        )

    def test_personal_shown_to_subject(self, memory_manager):
        """PERSONAL memories shown when querying user is the subject."""
        assert memory_manager._passes_privacy_filter(
            sensitivity=Sensitivity.PERSONAL,
            subject_person_ids=["person-1"],
            chat_type="group",
            querying_person_ids={"person-1"},
        )

    def test_personal_hidden_from_non_subject(self, memory_manager):
        """PERSONAL memories hidden when querying user is NOT the subject."""
        assert not memory_manager._passes_privacy_filter(
            sensitivity=Sensitivity.PERSONAL,
            subject_person_ids=["person-1"],
            chat_type="group",
            querying_person_ids={"person-2"},
        )

    def test_personal_hidden_when_no_person_ids(self, memory_manager):
        """PERSONAL memories hidden when no person IDs resolved."""
        assert not memory_manager._passes_privacy_filter(
            sensitivity=Sensitivity.PERSONAL,
            subject_person_ids=["person-1"],
            chat_type="private",
            querying_person_ids=set(),
        )

    def test_sensitive_shown_in_private_to_subject(self, memory_manager):
        """SENSITIVE memories shown only in private chat to subject."""
        assert memory_manager._passes_privacy_filter(
            sensitivity=Sensitivity.SENSITIVE,
            subject_person_ids=["person-1"],
            chat_type="private",
            querying_person_ids={"person-1"},
        )

    def test_sensitive_hidden_in_group_even_for_subject(self, memory_manager):
        """SENSITIVE memories hidden in group chat even for subject."""
        assert not memory_manager._passes_privacy_filter(
            sensitivity=Sensitivity.SENSITIVE,
            subject_person_ids=["person-1"],
            chat_type="group",
            querying_person_ids={"person-1"},
        )

    def test_sensitive_hidden_in_private_from_non_subject(self, memory_manager):
        """SENSITIVE memories hidden in private chat from non-subject."""
        assert not memory_manager._passes_privacy_filter(
            sensitivity=Sensitivity.SENSITIVE,
            subject_person_ids=["person-1"],
            chat_type="private",
            querying_person_ids={"person-2"},
        )


class TestOwnMemoryPrivacy:
    """Verify owner's own memories are never filtered by privacy rules.

    Regression: the privacy filter was applied to ALL memories including the
    owner's primary search results, incorrectly blocking SENSITIVE self-memories
    (empty subjects) and PERSONAL memories about others in group chats.
    """

    async def test_sensitive_self_memory_returned_in_group(
        self, ash_home, file_memory_store, mock_index, mock_embedding_generator
    ):
        """SENSITIVE self-memory (no subjects) visible in group chat."""
        memory = await file_memory_store.add_memory(
            content="I have been dealing with anxiety",
            owner_user_id="user-1",
            sensitivity=Sensitivity.SENSITIVE,
        )

        # Wire mock index to return this memory
        mock_index.search = AsyncMock(
            return_value=_make_search_results([(memory.id, 0.9)])
        )

        graph_dir = ash_home / "graph"
        graph_dir.mkdir(exist_ok=True)
        mm = GraphStore(
            memory_store=file_memory_store,
            vector_index=mock_index,
            embedding_generator=mock_embedding_generator,
            people_path=graph_dir / "people.jsonl",
            users_path=graph_dir / "users.jsonl",
            chats_path=graph_dir / "chats.jsonl",
        )

        ctx = await mm.get_context_for_message(
            user_id="user-1",
            user_message="how am I doing",
            chat_type="group",
            participant_person_ids={"dcramer": {"self-person-id"}},
        )

        assert len(ctx.memories) == 1
        assert ctx.memories[0].content == "I have been dealing with anxiety"

    async def test_personal_memory_about_other_returned_in_group(
        self, ash_home, file_memory_store, mock_index, mock_embedding_generator
    ):
        """PERSONAL memory about someone else visible to owner in group chat."""
        memory = await file_memory_store.add_memory(
            content="Sarah is going through a hard time",
            owner_user_id="user-1",
            sensitivity=Sensitivity.PERSONAL,
            subject_person_ids=["sarah-person-id"],
        )

        mock_index.search = AsyncMock(
            return_value=_make_search_results([(memory.id, 0.9)])
        )

        graph_dir = ash_home / "graph"
        graph_dir.mkdir(exist_ok=True)
        mm = GraphStore(
            memory_store=file_memory_store,
            vector_index=mock_index,
            embedding_generator=mock_embedding_generator,
            people_path=graph_dir / "people.jsonl",
            users_path=graph_dir / "users.jsonl",
            chats_path=graph_dir / "chats.jsonl",
        )

        ctx = await mm.get_context_for_message(
            user_id="user-1",
            user_message="how is Sarah",
            chat_type="group",
            participant_person_ids={"dcramer": {"self-person-id"}},
        )

        assert len(ctx.memories) == 1
        assert ctx.memories[0].content == "Sarah is going through a hard time"


class TestPortableExtraction:
    """Tests for portable field in extraction."""

    def test_extractor_parses_portable_false(self):
        """Extractor correctly parses portable=false from LLM output."""
        from ash.memory.extractor import MemoryExtractor

        extractor = MemoryExtractor.__new__(MemoryExtractor)
        extractor._confidence_threshold = 0.7

        response = '[{"content": "Bob is presenting next", "speaker": "david", "subjects": ["Bob"], "shared": true, "confidence": 0.9, "type": "context", "sensitivity": "public", "portable": false}]'
        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].portable is False

    def test_extractor_defaults_portable_true(self):
        """Extractor defaults portable to true when not specified."""
        from ash.memory.extractor import MemoryExtractor

        extractor = MemoryExtractor.__new__(MemoryExtractor)
        extractor._confidence_threshold = 0.7

        response = '[{"content": "Bob loves pizza", "speaker": "david", "subjects": ["Bob"], "shared": false, "confidence": 0.9, "type": "knowledge", "sensitivity": "public"}]'
        facts = extractor._parse_extraction_response(response)

        assert len(facts) == 1
        assert facts[0].portable is True


class TestForgetPerson:
    """Tests for forget-person operation."""

    async def test_forget_archives_subject_memories(
        self, memory_manager, file_memory_store, mock_index
    ):
        """Forget archives memories with ABOUT edges to the person."""
        person_id = "person-bob"

        # Memory about Bob
        about_bob = await file_memory_store.add_memory(
            content="Bob likes hiking",
            owner_user_id="alice",
            subject_person_ids=[person_id],
        )

        # Memory not about Bob
        unrelated = await file_memory_store.add_memory(
            content="Alice likes cooking",
            owner_user_id="alice",
        )

        archived_count = await memory_manager.forget_person(person_id=person_id)

        assert archived_count == 1
        assert await file_memory_store.get_memory(about_bob.id) is None
        assert await file_memory_store.get_memory(unrelated.id) is not None

        # Verify it was archived
        archived = await file_memory_store.get_archived_memories()
        assert len(archived) == 1
        assert archived[0].id == about_bob.id
        assert archived[0].archive_reason == "forgotten"

    async def test_forget_removes_from_vector_index(
        self, memory_manager, file_memory_store, mock_index
    ):
        """Forget removes embeddings from the vector index."""
        person_id = "person-bob"

        about_bob = await file_memory_store.add_memory(
            content="Bob likes hiking",
            owner_user_id="alice",
            subject_person_ids=[person_id],
        )

        await memory_manager.forget_person(person_id=person_id)

        mock_index.delete_embedding.assert_called_once_with(about_bob.id)

    async def test_forget_returns_zero_when_no_memories(
        self, memory_manager, file_memory_store
    ):
        """Forget returns 0 when person has no memories."""
        archived_count = await memory_manager.forget_person(person_id="nonexistent")

        assert archived_count == 0

    async def test_forget_with_delete_person_record(
        self, memory_manager, file_memory_store, mock_index
    ):
        """Forget can optionally delete the person record."""
        # Mock delete_person on the GraphStore instance
        memory_manager.delete_person = AsyncMock(return_value=True)

        person_id = "person-bob"
        await file_memory_store.add_memory(
            content="Bob likes hiking",
            owner_user_id="alice",
            subject_person_ids=[person_id],
        )

        await memory_manager.forget_person(
            person_id=person_id,
            delete_person_record=True,
        )

        memory_manager.delete_person.assert_called_once_with(person_id)


class TestSubjectNameResolution:
    """Verify subject_name is resolved in search result metadata."""

    async def test_search_resolves_subject_name(
        self, ash_home, file_memory_store, mock_index, mock_embedding_generator
    ):
        """Search results include subject_name when person records exist."""
        from ash.people.types import PersonEntry

        memory = await file_memory_store.add_memory(
            content="Sarah's birthday is March 15",
            owner_user_id="user-1",
            subject_person_ids=["sarah-id"],
        )

        mock_index.search = AsyncMock(
            return_value=_make_search_results([(memory.id, 0.9)])
        )

        graph_dir = ash_home / "graph"
        graph_dir.mkdir(exist_ok=True)
        mm = GraphStore(
            memory_store=file_memory_store,
            vector_index=mock_index,
            embedding_generator=mock_embedding_generator,
            people_path=graph_dir / "people.jsonl",
            users_path=graph_dir / "users.jsonl",
            chats_path=graph_dir / "chats.jsonl",
        )

        # Mock get_person to return a person entry with a name
        mock_person = MagicMock(spec=PersonEntry)
        mock_person.name = "Sarah"
        mm.get_person = AsyncMock(return_value=mock_person)  # type: ignore[assignment]

        results = await mm.search(query="birthday", owner_user_id="user-1")
        assert len(results) == 1
        assert results[0].metadata is not None
        assert results[0].metadata["subject_name"] == "Sarah"

    async def test_search_no_subject_name_without_people(
        self, ash_home, file_memory_store, mock_index, mock_embedding_generator
    ):
        """No subject_name when person records are not available."""
        memory = await file_memory_store.add_memory(
            content="Sarah's birthday is March 15",
            owner_user_id="user-1",
            subject_person_ids=["sarah-id"],
        )

        mock_index.search = AsyncMock(
            return_value=_make_search_results([(memory.id, 0.9)])
        )

        graph_dir = ash_home / "graph"
        graph_dir.mkdir(exist_ok=True)
        mm = GraphStore(
            memory_store=file_memory_store,
            vector_index=mock_index,
            embedding_generator=mock_embedding_generator,
            people_path=graph_dir / "people.jsonl",
            users_path=graph_dir / "users.jsonl",
            chats_path=graph_dir / "chats.jsonl",
        )

        results = await mm.search(query="birthday", owner_user_id="user-1")
        assert len(results) == 1
        assert results[0].metadata is not None
        assert "subject_name" not in results[0].metadata

    async def test_search_no_subject_name_for_self_facts(
        self, ash_home, file_memory_store, mock_index, mock_embedding_generator
    ):
        """No subject_name for facts about the speaker (empty subjects)."""
        memory = await file_memory_store.add_memory(
            content="Prefers dark mode",
            owner_user_id="user-1",
            subject_person_ids=[],
        )

        mock_index.search = AsyncMock(
            return_value=_make_search_results([(memory.id, 0.9)])
        )

        graph_dir = ash_home / "graph"
        graph_dir.mkdir(exist_ok=True)
        mm = GraphStore(
            memory_store=file_memory_store,
            vector_index=mock_index,
            embedding_generator=mock_embedding_generator,
            people_path=graph_dir / "people.jsonl",
            users_path=graph_dir / "users.jsonl",
            chats_path=graph_dir / "chats.jsonl",
        )

        results = await mm.search(query="dark mode", owner_user_id="user-1")
        assert len(results) == 1
        assert results[0].metadata is not None
        assert "subject_name" not in results[0].metadata
