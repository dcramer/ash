"""Tests for memory system behavior.

Tests focus on:
- Supersession logic (core business behavior)
- Garbage collection and eviction
- Scoping rules (personal vs group)
- Error handling

Note: These tests use the in-memory KnowledgeGraph-backed Store architecture.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.graph.graph import KnowledgeGraph
from ash.graph.persistence import GraphPersistence
from ash.memory.embeddings import EmbeddingGenerator
from ash.store.store import Store
from ash.store.types import MemoryType, Sensitivity


def _make_search_results(
    results: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Convert (memory_id, similarity) tuples — identity pass-through."""
    return results


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
    index = MagicMock()
    index.search = MagicMock(return_value=[])
    index.add = MagicMock()
    index.remove = MagicMock()
    index.save = AsyncMock()
    index.get_ids = MagicMock(return_value=set())
    return index


@pytest.fixture
async def graph_store(graph_dir, mock_index, mock_embedding_generator) -> Store:
    """Create a Store with mocked components."""
    graph = KnowledgeGraph()
    persistence = GraphPersistence(graph_dir)
    store = Store(
        graph=graph,
        persistence=persistence,
        vector_index=mock_index,
        embedding_generator=mock_embedding_generator,
    )
    store._llm_model = "mock-model"
    return store


class TestMemorySupersession:
    """Tests for memory supersession - core business logic."""

    async def test_mark_memory_superseded(self, graph_store: Store):
        """Test marking a memory as superseded."""
        old_memory = await graph_store.add_memory(
            content="User's favorite color is red",
            owner_user_id="user-1",
        )
        new_memory = await graph_store.add_memory(
            content="User's favorite color is blue",
            owner_user_id="user-1",
        )

        result = await graph_store.batch_mark_superseded(
            [(old_memory.id, new_memory.id)]
        )

        assert len(result) == 1
        # get_memory excludes archived/superseded; use graph directly to verify
        mem = graph_store.graph.memories[old_memory.id]
        assert mem.superseded_at is not None
        # Verify via SUPERSEDES edge
        from ash.graph.edges import get_superseded_by

        assert get_superseded_by(graph_store.graph, old_memory.id) == new_memory.id

    async def test_list_memories_excludes_superseded_by_default(
        self, graph_store: Store
    ):
        """Test that list_memories excludes superseded memories."""
        old_memory = await graph_store.add_memory(content="Old fact")
        new_memory = await graph_store.add_memory(content="New fact")
        await graph_store.batch_mark_superseded([(old_memory.id, new_memory.id)])

        memories = await graph_store.list_memories(include_superseded=False)

        assert len(memories) == 1
        assert memories[0].content == "New fact"

    async def test_list_memories_can_include_superseded(self, graph_store: Store):
        """Test that list_memories can include superseded memories."""
        old_memory = await graph_store.add_memory(content="Old fact")
        new_memory = await graph_store.add_memory(content="New fact")
        await graph_store.batch_mark_superseded([(old_memory.id, new_memory.id)])

        memories = await graph_store.list_memories(include_superseded=True)

        assert len(memories) == 2


class TestStoreSupersession:
    """Tests for automatic supersession via Store."""

    async def test_add_memory_supersedes_conflicting(
        self, graph_store: Store, mock_index
    ):
        """Test that adding a memory supersedes high-similarity conflicts."""
        old_memory = await graph_store.add_memory(
            content="User's favorite color is red",
            owner_user_id="user-1",
        )

        # Mock the index to return the old memory as similar
        mock_index.search.return_value = _make_search_results(
            [
                (old_memory.id, 0.85)  # Above 0.75 threshold
            ]
        )

        new_memory = await graph_store.add_memory(
            content="User's favorite color is blue",
            owner_user_id="user-1",
        )

        # Check that old memory is now superseded via graph
        mem = graph_store.graph.memories[old_memory.id]
        assert mem.superseded_at is not None
        # Verify via SUPERSEDES edge
        from ash.graph.edges import get_superseded_by

        assert get_superseded_by(graph_store.graph, old_memory.id) == new_memory.id

    async def test_no_supersession_below_threshold(
        self, graph_store: Store, mock_index
    ):
        """Test that memories below similarity threshold are not superseded."""
        old_memory = await graph_store.add_memory(
            content="User likes pizza",
            owner_user_id="user-1",
        )

        # Mock the index to return low similarity
        mock_index.search.return_value = _make_search_results(
            [
                (old_memory.id, 0.5)  # Below 0.75 threshold
            ]
        )

        await graph_store.add_memory(
            content="User likes coffee",
            owner_user_id="user-1",
        )

        # Old memory should NOT be superseded
        mem = graph_store.graph.memories[old_memory.id]
        assert mem.superseded_at is None

    async def test_batch_update_reindexes_when_content_changes(
        self,
        graph_store: Store,
        mock_index: MagicMock,
        mock_embedding_generator: MagicMock,
    ) -> None:
        memory = await graph_store.add_memory(content="Old value", owner_user_id="u1")
        mock_index.reset_mock()
        mock_embedding_generator.embed.reset_mock()

        updated = memory.model_copy(deep=True)
        updated.content = "New value"

        await graph_store.batch_update_memories([updated])

        mock_embedding_generator.embed.assert_awaited_once_with("New value")
        mock_index.remove.assert_called_once_with(memory.id)
        mock_index.add.assert_called_once()
        assert graph_store.graph.memories[memory.id].content == "New value"

    async def test_batch_update_skips_reindex_when_content_unchanged(
        self,
        graph_store: Store,
        mock_index: MagicMock,
        mock_embedding_generator: MagicMock,
    ) -> None:
        memory = await graph_store.add_memory(
            content="Stable value", owner_user_id="u1"
        )
        mock_index.reset_mock()
        mock_embedding_generator.embed.reset_mock()

        updated = memory.model_copy(deep=True)
        updated.source_display_name = "Alice"

        await graph_store.batch_update_memories([updated])

        mock_embedding_generator.embed.assert_not_called()
        mock_index.remove.assert_not_called()
        mock_index.add.assert_not_called()


class TestGarbageCollection:
    """Tests for memory garbage collection."""

    async def test_gc_removes_expired_memories(self, graph_store: Store):
        """Test that GC removes expired memories."""
        past = datetime.now(UTC) - timedelta(days=1)
        expired = await graph_store.add_memory(content="Expired", expires_at=past)
        valid = await graph_store.add_memory(content="Valid")

        result = await graph_store.gc()

        assert result.removed_count == 1
        assert expired.id in result.archived_ids
        assert await graph_store.get_memory(expired.id) is None
        assert await graph_store.get_memory(valid.id) is not None

    async def test_gc_removes_superseded_memories(self, graph_store: Store):
        """Test that GC removes superseded memories."""
        old = await graph_store.add_memory(content="Old fact")
        new = await graph_store.add_memory(content="New fact")
        await graph_store.batch_mark_superseded([(old.id, new.id)])

        result = await graph_store.gc()

        assert result.removed_count == 1
        assert old.id in result.archived_ids
        assert await graph_store.get_memory(old.id) is None
        assert await graph_store.get_memory(new.id) is not None

    async def test_gc_handles_empty_state(self, graph_store: Store):
        """Test that GC works with only valid memories."""
        await graph_store.add_memory(content="Valid 1")
        await graph_store.add_memory(content="Valid 2")

        result = await graph_store.gc()

        assert result.removed_count == 0


class TestEnforceMaxEntries:
    """Tests for max_entries eviction policy."""

    async def test_evicts_oldest_when_over_limit(self, graph_store: Store):
        """Test that enforce_max_entries evicts oldest memories."""
        old_time = datetime.now(UTC) - timedelta(days=10)
        memories = []
        for i in range(5):
            m = await graph_store.add_memory(content=f"Fact {i}")
            # Adjust created_at via in-memory graph to stagger them
            graph_store.graph.memories[m.id].created_at = old_time + timedelta(hours=i)
            memories.append(m)

        evicted = await graph_store.enforce_max_entries(3)

        assert evicted == 2
        remaining = await graph_store.list_memories(limit=None)
        assert len(remaining) == 3

    async def test_no_eviction_when_under_limit(self, graph_store: Store):
        """Test that no eviction happens when under limit."""
        await graph_store.add_memory(content="Fact 1")
        await graph_store.add_memory(content="Fact 2")

        evicted = await graph_store.enforce_max_entries(10)

        assert evicted == 0


class TestScoping:
    """Tests for memory scoping (personal vs group)."""

    async def test_personal_scope_filters_by_user(self, graph_store: Store):
        """Test that personal memories are filtered by user."""
        await graph_store.add_memory(content="User 1 fact", owner_user_id="user-1")
        await graph_store.add_memory(content="User 2 fact", owner_user_id="user-2")

        user1_memories = await graph_store.list_memories(
            owner_user_id="user-1", limit=None
        )

        assert len(user1_memories) == 1
        assert user1_memories[0].content == "User 1 fact"

    async def test_group_scope_filters_by_chat(self, graph_store: Store):
        """Test that group memories are filtered by chat."""
        await graph_store.add_memory(content="Chat 1 fact", chat_id="chat-1")
        await graph_store.add_memory(content="Chat 2 fact", chat_id="chat-2")

        chat1_memories = await graph_store.list_memories(chat_id="chat-1", limit=None)

        assert len(chat1_memories) == 1
        assert chat1_memories[0].content == "Chat 1 fact"

    async def test_list_memories_limit_respects_scope(self, graph_store: Store):
        """LIMIT should apply after scope filtering, returning correct count."""
        # Create 10 memories for user-1 and 10 for user-2
        for i in range(10):
            await graph_store.add_memory(
                content=f"User1 fact {i}", owner_user_id="user-1"
            )
            await graph_store.add_memory(
                content=f"User2 fact {i}", owner_user_id="user-2"
            )

        # Ask for 5 memories scoped to user-1
        user1_memories = await graph_store.list_memories(
            owner_user_id="user-1", limit=5
        )

        assert len(user1_memories) == 5
        # All should belong to user-1
        for m in user1_memories:
            assert m.owner_user_id == "user-1"

    async def test_list_memories_returns_detached_entries(self, graph_store: Store):
        """Mutating list results should not mutate graph state."""
        memory = await graph_store.add_memory(
            content="Detached", owner_user_id="user-1"
        )

        memories = await graph_store.list_memories(owner_user_id="user-1", limit=None)
        assert len(memories) == 1
        memories[0].content = "Mutated copy"

        assert graph_store.graph.memories[memory.id].content == "Detached"

    async def test_get_memory_returns_detached_entry(self, graph_store: Store):
        """Mutating get_memory result should not mutate graph state."""
        memory = await graph_store.add_memory(content="Original")
        loaded = await graph_store.get_memory(memory.id)
        assert loaded is not None

        loaded.content = "Mutated copy"
        assert graph_store.graph.memories[memory.id].content == "Original"


class TestEphemeralDecay:
    """Tests for ephemeral memory type decay."""

    async def test_observation_decays_after_ttl(self, graph_store: Store):
        """Test that observation memories decay after 3 days."""
        old_time = datetime.now(UTC) - timedelta(days=5)
        observation = await graph_store.add_memory(
            content="Noticed something",
            memory_type=MemoryType.OBSERVATION,
        )
        # Adjust created_at via in-memory graph
        graph_store.graph.memories[observation.id].created_at = old_time

        result = await graph_store.gc()

        assert result.removed_count == 1
        assert observation.id in result.archived_ids

    async def test_preference_never_decays(self, graph_store: Store):
        """Test that preference memories never decay."""
        old_time = datetime.now(UTC) - timedelta(days=365)
        preference = await graph_store.add_memory(
            content="Likes dark mode",
            memory_type=MemoryType.PREFERENCE,
        )
        # Adjust created_at via in-memory graph
        graph_store.graph.memories[preference.id].created_at = old_time

        result = await graph_store.gc()

        assert result.removed_count == 0
        assert await graph_store.get_memory(preference.id) is not None


class TestCrossContextRetrieval:
    """Tests for cross-context memory retrieval using person IDs."""

    async def test_find_memories_by_subject(self, graph_store: Store):
        """Test finding memories by subject person IDs."""
        from ash.store.retrieval import RetrievalPipeline

        person_id = "person-bob-1"
        pipeline = RetrievalPipeline(graph_store)

        # Create a memory about this person
        await graph_store.add_memory(
            content="Bob likes pizza",
            owner_user_id="alice",
            subject_person_ids=[person_id],
        )

        # Also a memory not about Bob
        await graph_store.add_memory(
            content="General fact",
            owner_user_id="alice",
        )

        memories = await pipeline._find_memories_about_persons(person_ids={person_id})

        assert len(memories) == 1
        assert memories[0].content == "Bob likes pizza"

    async def test_find_memories_by_subject_excludes_owner(self, graph_store: Store):
        """Test that _find_memories_about_persons can exclude owner's memories."""
        from ash.store.retrieval import RetrievalPipeline

        person_id_alice = "person-bob-alice"
        person_id_carol = "person-bob-carol"
        pipeline = RetrievalPipeline(graph_store)

        # Alice has a memory about Bob
        await graph_store.add_memory(
            content="Bob likes pizza",
            owner_user_id="alice",
            subject_person_ids=[person_id_alice],
        )

        # Carol also has a memory about Bob
        await graph_store.add_memory(
            content="Bob likes pasta",
            owner_user_id="carol",
            subject_person_ids=[person_id_carol],
        )

        # Find memories about bob excluding alice's
        memories = await pipeline._find_memories_about_persons(
            person_ids={person_id_alice, person_id_carol},
            exclude_owner_user_id="alice",
        )

        assert len(memories) == 1
        assert memories[0].content == "Bob likes pasta"
        assert memories[0].owner_user_id == "carol"

    async def test_find_memories_by_subject_multiple_ids(self, graph_store: Store):
        """Test that multiple person IDs find all relevant memories."""
        from ash.store.retrieval import RetrievalPipeline

        person_id_1 = "person-bob-1"
        person_id_2 = "person-bob-2"
        pipeline = RetrievalPipeline(graph_store)

        await graph_store.add_memory(
            content="Bob fact from user1",
            owner_user_id="user1",
            subject_person_ids=[person_id_1],
        )

        await graph_store.add_memory(
            content="Bob fact from user2",
            owner_user_id="user2",
            subject_person_ids=[person_id_2],
        )

        memories = await pipeline._find_memories_about_persons(
            person_ids={person_id_1, person_id_2}
        )

        assert len(memories) == 2


class TestSecretsFiltering:
    """Tests for secrets filtering in Store."""

    async def test_rejects_openai_api_key(self, graph_store: Store):
        """Test that OpenAI API keys are rejected."""
        with pytest.raises(ValueError, match="potential secrets"):
            await graph_store.add_memory(
                content="My API key is sk-abc123def456ghi789abcdef",
                owner_user_id="user-1",
            )

    async def test_rejects_github_token(self, graph_store: Store):
        """Test that GitHub tokens are rejected."""
        with pytest.raises(ValueError, match="potential secrets"):
            await graph_store.add_memory(
                content="Use token: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                owner_user_id="user-1",
            )

    async def test_rejects_aws_access_key(self, graph_store: Store):
        """Test that AWS access keys are rejected."""
        with pytest.raises(ValueError, match="potential secrets"):
            await graph_store.add_memory(
                content="AWS key: AKIAIOSFODNN7EXAMPLE",
                owner_user_id="user-1",
            )

    async def test_rejects_credit_card(self, graph_store: Store):
        """Test that credit card numbers are rejected."""
        with pytest.raises(ValueError, match="potential secrets"):
            await graph_store.add_memory(
                content="Card: 4111-1111-1111-1111",
                owner_user_id="user-1",
            )

    async def test_rejects_ssn(self, graph_store: Store):
        """Test that Social Security Numbers are rejected."""
        with pytest.raises(ValueError, match="potential secrets"):
            await graph_store.add_memory(
                content="SSN: 123-45-6789",
                owner_user_id="user-1",
            )

    async def test_rejects_password_is_pattern(self, graph_store: Store):
        """Test that 'password is X' patterns are rejected."""
        with pytest.raises(ValueError, match="potential secrets"):
            await graph_store.add_memory(
                content="Remember my password is hunter2",
                owner_user_id="user-1",
            )

    async def test_rejects_password_colon_pattern(self, graph_store: Store):
        """Test that 'password: X' patterns are rejected."""
        with pytest.raises(ValueError, match="potential secrets"):
            await graph_store.add_memory(
                content="My password: secret123",
                owner_user_id="user-1",
            )

    async def test_rejects_private_key(self, graph_store: Store):
        """Test that private keys are rejected."""
        with pytest.raises(ValueError, match="potential secrets"):
            await graph_store.add_memory(
                content="-----BEGIN PRIVATE KEY-----\nMIIEvg...",
                owner_user_id="user-1",
            )

    async def test_allows_safe_content(self, graph_store: Store):
        """Test that normal content is allowed."""
        memory = await graph_store.add_memory(
            content="User prefers dark mode",
            owner_user_id="user-1",
        )
        assert memory.content == "User prefers dark mode"

    async def test_allows_number_sequences(self, graph_store: Store):
        """Test that normal number sequences are allowed."""
        # 4 digits not credit card, no SSN pattern
        memory = await graph_store.add_memory(
            content="Favorite number is 1234",
            owner_user_id="user-1",
        )
        assert memory.content == "Favorite number is 1234"


class TestSensitivity:
    """Tests for memory sensitivity classification."""

    async def test_add_memory_with_sensitivity(self, graph_store: Store):
        """Test adding a memory with sensitivity classification."""
        memory = await graph_store.add_memory(
            content="Has anxiety",
            owner_user_id="user-1",
            sensitivity=Sensitivity.SENSITIVE,
        )

        assert memory.sensitivity == Sensitivity.SENSITIVE

        # Verify it persists
        loaded = await graph_store.get_memory(memory.id)
        assert loaded is not None
        assert loaded.sensitivity == Sensitivity.SENSITIVE

    async def test_default_sensitivity_is_public(self, graph_store: Store):
        """Test that omitted sensitivity defaults to PUBLIC."""
        memory = await graph_store.add_memory(
            content="Likes pizza",
            owner_user_id="user-1",
        )

        assert memory.sensitivity == Sensitivity.PUBLIC

    async def test_memory_serialization_with_sensitivity(self, graph_store: Store):
        """Test that sensitivity is correctly stored and retrieved."""
        memory = await graph_store.add_memory(
            content="Personal info",
            owner_user_id="user-1",
            sensitivity=Sensitivity.PERSONAL,
        )

        loaded = await graph_store.get_memory(memory.id)
        assert loaded is not None
        assert loaded.sensitivity == Sensitivity.PERSONAL


class TestPrivacyFilter:
    """Regression tests for privacy filter bugs.

    Bug 1: _is_user_subject_of_memory always returned False (now fixed by
    accepting pre-resolved person IDs instead of doing sync lookups).

    Bug 2: Inconsistent filter logic between MemoryEntry and SearchResult
    paths (now unified into shared visibility policy helpers).
    """

    def test_public_always_passes(self):
        """PUBLIC memories pass regardless of context."""
        from ash.store.visibility import passes_sensitivity_policy

        assert passes_sensitivity_policy(
            sensitivity=Sensitivity.PUBLIC,
            subject_person_ids=["person-1"],
            chat_type="group",
            querying_person_ids=set(),
        )

    def test_public_sensitivity_treated_as_public(self):
        """PUBLIC sensitivity passes."""
        from ash.store.visibility import passes_sensitivity_policy

        assert passes_sensitivity_policy(
            sensitivity=Sensitivity.PUBLIC,
            subject_person_ids=["person-1"],
            chat_type="group",
            querying_person_ids=set(),
        )

    def test_personal_shown_to_subject(self):
        """PERSONAL memories shown when querying user is the subject."""
        from ash.store.visibility import passes_sensitivity_policy

        assert passes_sensitivity_policy(
            sensitivity=Sensitivity.PERSONAL,
            subject_person_ids=["person-1"],
            chat_type="group",
            querying_person_ids={"person-1"},
        )

    def test_personal_hidden_from_non_subject(self):
        """PERSONAL memories hidden when querying user is NOT the subject."""
        from ash.store.visibility import passes_sensitivity_policy

        assert not passes_sensitivity_policy(
            sensitivity=Sensitivity.PERSONAL,
            subject_person_ids=["person-1"],
            chat_type="group",
            querying_person_ids={"person-2"},
        )

    def test_personal_hidden_when_no_person_ids(self):
        """PERSONAL memories hidden when no person IDs resolved."""
        from ash.store.visibility import passes_sensitivity_policy

        assert not passes_sensitivity_policy(
            sensitivity=Sensitivity.PERSONAL,
            subject_person_ids=["person-1"],
            chat_type="private",
            querying_person_ids=set(),
        )

    def test_sensitive_shown_in_private_to_subject(self):
        """SENSITIVE memories shown only in private chat to subject."""
        from ash.store.visibility import passes_sensitivity_policy

        assert passes_sensitivity_policy(
            sensitivity=Sensitivity.SENSITIVE,
            subject_person_ids=["person-1"],
            chat_type="private",
            querying_person_ids={"person-1"},
        )

    def test_sensitive_hidden_in_group_even_for_subject(self):
        """SENSITIVE memories hidden in group chat even for subject."""
        from ash.store.visibility import passes_sensitivity_policy

        assert not passes_sensitivity_policy(
            sensitivity=Sensitivity.SENSITIVE,
            subject_person_ids=["person-1"],
            chat_type="group",
            querying_person_ids={"person-1"},
        )

    def test_sensitive_hidden_in_private_from_non_subject(self):
        """SENSITIVE memories hidden in private chat from non-subject."""
        from ash.store.visibility import passes_sensitivity_policy

        assert not passes_sensitivity_policy(
            sensitivity=Sensitivity.SENSITIVE,
            subject_person_ids=["person-1"],
            chat_type="private",
            querying_person_ids={"person-2"},
        )


class TestOwnMemoryPrivacy:
    """Verify owner's self-memories and personal notes survive Stage 1 filtering.

    Stage 1 privacy filtering targets SENSITIVE memories about other people
    in group chats (health, medical, financial). Self-memories (no subjects)
    and PERSONAL notes the owner wrote always pass through.
    """

    async def test_sensitive_self_memory_returned_in_group(
        self, graph_store: Store, mock_index
    ):
        """SENSITIVE self-memory (no subjects) visible in group chat."""
        from ash.graph.edges import create_learned_in_edge
        from ash.store.types import ChatEntry

        memory = await graph_store.add_memory(
            content="I have been dealing with anxiety",
            owner_user_id="user-1",
            sensitivity=Sensitivity.SENSITIVE,
        )
        graph_store.graph.add_chat(
            ChatEntry(
                id="group-chat-1",
                provider="telegram",
                provider_id="group-1",
                chat_type="group",
            )
        )
        graph_store.graph.add_edge(create_learned_in_edge(memory.id, "group-chat-1"))

        # Wire mock index to return this memory
        mock_index.search.return_value = [(memory.id, 0.9)]

        ctx = await graph_store.get_context_for_message(
            user_id="user-1",
            user_message="how am I doing",
            chat_type="group",
            participant_person_ids={"dcramer": {"self-person-id"}},
        )

        assert len(ctx.memories) == 1
        assert ctx.memories[0].content == "I have been dealing with anxiety"

    async def test_personal_memory_about_non_participant_excluded_in_group(
        self, graph_store: Store, mock_index
    ):
        """PERSONAL memory about a non-participant excluded in group chat."""
        from ash.graph.edges import create_learned_in_edge
        from ash.store.types import ChatEntry

        memory = await graph_store.add_memory(
            content="Sarah is going through a hard time",
            owner_user_id="user-1",
            sensitivity=Sensitivity.PERSONAL,
            subject_person_ids=["sarah-person-id"],
        )
        graph_store.graph.add_chat(
            ChatEntry(
                id="group-chat-1",
                provider="telegram",
                provider_id="group-1",
                chat_type="group",
            )
        )
        graph_store.graph.add_edge(create_learned_in_edge(memory.id, "group-chat-1"))

        mock_index.search.return_value = [(memory.id, 0.9)]

        ctx = await graph_store.get_context_for_message(
            user_id="user-1",
            user_message="how is Sarah",
            chat_type="group",
            participant_person_ids={"dcramer": {"self-person-id"}},
        )

        assert len(ctx.memories) == 0

    async def test_personal_memory_about_participant_shown_in_group(
        self, graph_store: Store, mock_index
    ):
        """PERSONAL memory about a participant shown in group chat."""
        from ash.graph.edges import create_learned_in_edge
        from ash.store.types import ChatEntry

        memory = await graph_store.add_memory(
            content="Sarah is going through a hard time",
            owner_user_id="user-1",
            sensitivity=Sensitivity.PERSONAL,
            subject_person_ids=["sarah-person-id"],
        )
        graph_store.graph.add_chat(
            ChatEntry(
                id="group-chat-1",
                provider="telegram",
                provider_id="group-1",
                chat_type="group",
            )
        )
        graph_store.graph.add_edge(create_learned_in_edge(memory.id, "group-chat-1"))

        mock_index.search.return_value = [(memory.id, 0.9)]

        ctx = await graph_store.get_context_for_message(
            user_id="user-1",
            user_message="how is Sarah",
            chat_type="group",
            participant_person_ids={"sarah": {"sarah-person-id"}},
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
        self, graph_store: Store, mock_index
    ):
        """Forget archives memories with ABOUT edges to the person."""
        person_id = "person-bob"

        # Memory about Bob
        about_bob = await graph_store.add_memory(
            content="Bob likes hiking",
            owner_user_id="alice",
            subject_person_ids=[person_id],
        )

        # Memory not about Bob
        unrelated = await graph_store.add_memory(
            content="Alice likes cooking",
            owner_user_id="alice",
        )

        archived_count = await graph_store.forget_person(person_id=person_id)

        assert archived_count == 1
        assert await graph_store.get_memory(about_bob.id) is None
        assert await graph_store.get_memory(unrelated.id) is not None

        # Verify it was archived
        all_memories = await graph_store.get_all_memories()
        archived = [m for m in all_memories if m.archived_at is not None]
        assert len(archived) == 1
        assert archived[0].id == about_bob.id
        assert archived[0].archive_reason == "forgotten"

    async def test_forget_removes_from_vector_index(
        self, graph_store: Store, mock_index
    ):
        """Forget removes embeddings from the vector index."""
        person_id = "person-bob"

        about_bob = await graph_store.add_memory(
            content="Bob likes hiking",
            owner_user_id="alice",
            subject_person_ids=[person_id],
        )

        await graph_store.forget_person(person_id=person_id)

        mock_index.remove.assert_called_once_with(about_bob.id)

    async def test_forget_returns_zero_when_no_memories(self, graph_store: Store):
        """Forget returns 0 when person has no memories."""
        archived_count = await graph_store.forget_person(person_id="nonexistent")

        assert archived_count == 0

    async def test_forget_with_delete_person_record(
        self, graph_store: Store, mock_index
    ):
        """Forget can optionally delete the person record."""
        # Mock delete_person on the Store instance
        graph_store.delete_person = AsyncMock(return_value=True)  # type: ignore[assignment]

        person_id = "person-bob"
        await graph_store.add_memory(
            content="Bob likes hiking",
            owner_user_id="alice",
            subject_person_ids=[person_id],
        )

        await graph_store.forget_person(
            person_id=person_id,
            delete_person_record=True,
        )

        graph_store.delete_person.assert_called_once_with(person_id)  # type: ignore[union-attr]


class TestSubjectNameResolution:
    """Verify subject_name is resolved in search result metadata."""

    async def test_search_resolves_subject_name(self, graph_store: Store, mock_index):
        """Search results include subject_name when person records exist."""
        from ash.store.types import PersonEntry

        memory = await graph_store.add_memory(
            content="Sarah's birthday is March 15",
            owner_user_id="user-1",
            subject_person_ids=["sarah-id"],
        )

        mock_index.search.return_value = [(memory.id, 0.9)]

        # Mock get_person to return a person entry with a name
        mock_person = MagicMock(spec=PersonEntry)
        mock_person.name = "Sarah"
        graph_store.get_person = AsyncMock(return_value=mock_person)  # type: ignore[assignment]

        results = await graph_store.search(query="birthday", owner_user_id="user-1")
        assert len(results) == 1
        assert results[0].metadata is not None
        assert results[0].metadata["subject_name"] == "Sarah"

    async def test_search_no_subject_name_without_people(
        self, graph_store: Store, mock_index
    ):
        """No subject_name when person records are not available."""
        memory = await graph_store.add_memory(
            content="Sarah's birthday is March 15",
            owner_user_id="user-1",
            subject_person_ids=["sarah-id"],
        )

        mock_index.search.return_value = [(memory.id, 0.9)]

        results = await graph_store.search(query="birthday", owner_user_id="user-1")
        assert len(results) == 1
        assert results[0].metadata is not None
        assert "subject_name" not in results[0].metadata

    async def test_search_no_subject_name_for_self_facts(
        self, graph_store: Store, mock_index
    ):
        """No subject_name for facts about the speaker (empty subjects)."""
        memory = await graph_store.add_memory(
            content="Prefers dark mode",
            owner_user_id="user-1",
            subject_person_ids=[],
        )

        mock_index.search.return_value = [(memory.id, 0.9)]

        results = await graph_store.search(query="dark mode", owner_user_id="user-1")
        assert len(results) == 1
        assert results[0].metadata is not None
        assert "subject_name" not in results[0].metadata


class TestHearsaySupersession:
    """Tests for hearsay supersession when self-facts confirm hearsay."""

    async def test_self_fact_supersedes_hearsay(self, graph_store: Store, mock_index):
        """Store hearsay about Bob from Alice, then Bob's self-fact supersedes it."""
        # Create a person record for Bob and link username
        person = await graph_store.create_person(
            created_by="alice", name="Bob", aliases=["bob"]
        )
        # Link bob username -> person via ensure_user
        await graph_store.ensure_user(
            provider="test",
            provider_id="test",
            username="bob",
            person_id=person.id,
        )

        # Alice says something about Bob (hearsay)
        hearsay = await graph_store.add_memory(
            content="Bob likes pasta",
            owner_user_id="alice",
            source_username="alice",
            subject_person_ids=[person.id],
        )

        # Mock index to return the hearsay as similar to Bob's self-fact
        mock_index.search.return_value = [(hearsay.id, 0.90)]

        # Bob confirms directly (self-fact)
        from ash.store.hearsay import supersede_hearsay_for_fact

        self_fact = await graph_store.add_memory(
            content="I like pasta",
            owner_user_id="alice",
            source_username="bob",
            subject_person_ids=[],
        )

        count = await supersede_hearsay_for_fact(
            store=graph_store,
            new_memory=self_fact,
            source_username="bob",
        )

        # Depending on pipeline ordering, hearsay may already be superseded during
        # add_memory() before this explicit helper runs. The helper is idempotent.
        assert count in (0, 1)

        # Verify hearsay is now superseded via graph
        mem = graph_store.graph.memories[hearsay.id]
        assert mem.superseded_at is not None
        # Verify via SUPERSEDES edge
        from ash.graph.edges import get_superseded_by

        assert get_superseded_by(graph_store.graph, hearsay.id) == self_fact.id


class TestOwnerFilteringInProcessing:
    """Tests for owner filtering in process_extracted_facts.

    Ensures joint facts (owner + others) keep the owner as a subject,
    while pure self-facts (owner is sole subject) strip the owner.
    """

    @pytest.mark.asyncio
    async def test_joint_fact_keeps_owner_and_non_owner_subjects(
        self, graph_store: Store
    ):
        """When owner is one of multiple subjects, both should be kept."""
        from ash.memory.processing import process_extracted_facts
        from ash.store.types import ExtractedFact

        # Create the other person first so resolve_or_create_person can find them
        alice = await graph_store.create_person(created_by="user-1", name="Alice")

        facts = [
            ExtractedFact(
                content="David and Alice are starting a company together",
                subjects=["David", "Alice"],
                shared=False,
                confidence=0.9,
            )
        ]
        stored_ids = await process_extracted_facts(
            facts=facts,
            store=graph_store,
            user_id="user-1",
            speaker_username="david",
            speaker_display_name="David Cramer",
            owner_names=["david", "David Cramer"],
        )

        assert len(stored_ids) == 1
        mem = graph_store.graph.memories[stored_ids[0]]
        from ash.graph.edges import get_subject_person_ids

        subject_pids = get_subject_person_ids(graph_store.graph, mem.id)
        # Both David (resolved as a person) and Alice should be subjects
        assert len(subject_pids) >= 2
        assert alice.id in subject_pids

    @pytest.mark.asyncio
    async def test_pure_self_fact_strips_owner(self, graph_store: Store):
        """When owner is the sole subject, it should be stripped (self-fact injection handles it)."""
        from ash.memory.processing import process_extracted_facts
        from ash.store.types import ExtractedFact

        facts = [
            ExtractedFact(
                content="David likes pizza",
                subjects=["David"],
                shared=False,
                confidence=0.9,
            )
        ]
        stored_ids = await process_extracted_facts(
            facts=facts,
            store=graph_store,
            user_id="user-1",
            speaker_username="david",
            speaker_display_name="David Cramer",
            speaker_person_id="speaker-pid-1",
            owner_names=["david", "David Cramer"],
        )

        assert len(stored_ids) == 1
        from ash.graph.edges import get_subject_person_ids

        subject_pids = get_subject_person_ids(graph_store.graph, stored_ids[0])
        # Self-fact injection should set speaker_person_id, not the owner as a resolved person
        assert subject_pids == ["speaker-pid-1"]

    @pytest.mark.asyncio
    async def test_non_owner_subjects_unaffected(self, graph_store: Store):
        """Non-owner subjects should always be resolved normally."""
        from ash.memory.processing import process_extracted_facts
        from ash.store.types import ExtractedFact

        bob = await graph_store.create_person(created_by="user-1", name="Bob")

        facts = [
            ExtractedFact(
                content="Bob got promoted at work",
                subjects=["Bob"],
                shared=False,
                confidence=0.9,
            )
        ]
        stored_ids = await process_extracted_facts(
            facts=facts,
            store=graph_store,
            user_id="user-1",
            speaker_username="david",
            speaker_display_name="David Cramer",
            owner_names=["david", "David Cramer"],
        )

        assert len(stored_ids) == 1
        from ash.graph.edges import get_subject_person_ids

        subject_pids = get_subject_person_ids(graph_store.graph, stored_ids[0])
        assert bob.id in subject_pids

    @pytest.mark.asyncio
    async def test_speaker_lookup_is_deterministic_when_multiple_ids(
        self, graph_store: Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Speaker ID fallback should choose deterministically across set order."""
        from ash.memory.processing import process_extracted_facts
        from ash.store.types import ExtractedFact

        async def fake_find_person_ids_for_username(_username: str) -> set[str]:
            return {"pid-z", "pid-a"}

        monkeypatch.setattr(
            graph_store,
            "find_person_ids_for_username",
            fake_find_person_ids_for_username,
        )

        stored_ids = await process_extracted_facts(
            facts=[
                ExtractedFact(
                    content="I prefer tea",
                    subjects=[],
                    shared=False,
                    confidence=0.9,
                    speaker="dave",
                )
            ],
            store=graph_store,
            user_id="user-1",
        )

        assert len(stored_ids) == 1
        memory = graph_store.graph.memories[stored_ids[0]]
        assert memory.metadata is not None
        assertion = memory.metadata["assertion"]
        assert assertion["speaker_person_id"] == "pid-a"


class TestDMSensitivityFloor:
    """Tests for DM sensitivity floor on ephemeral memory types."""

    @pytest.mark.asyncio
    async def test_dm_event_gets_personal_sensitivity(self, graph_store: Store):
        """Ephemeral event memory extracted in DM gets minimum PERSONAL sensitivity."""
        from ash.memory.processing import process_extracted_facts
        from ash.store.types import ExtractedFact, MemoryType

        facts = [
            ExtractedFact(
                content="Planning a baby moon trip",
                subjects=[],
                shared=False,
                confidence=0.9,
                memory_type=MemoryType.EVENT,
                sensitivity=None,  # Would be PUBLIC by default
            ),
        ]

        stored_ids = await process_extracted_facts(
            facts=facts,
            store=graph_store,
            user_id="user-1",
            chat_type="private",
        )

        assert len(stored_ids) == 1
        mem = await graph_store.get_memory(stored_ids[0])
        assert mem is not None
        assert mem.sensitivity == Sensitivity.PERSONAL

    @pytest.mark.asyncio
    async def test_dm_preference_keeps_original_sensitivity(self, graph_store: Store):
        """Non-ephemeral preference type keeps original sensitivity in DM."""
        from ash.memory.processing import process_extracted_facts
        from ash.store.types import ExtractedFact, MemoryType

        facts = [
            ExtractedFact(
                content="Likes Italian food",
                subjects=[],
                shared=False,
                confidence=0.9,
                memory_type=MemoryType.PREFERENCE,
                sensitivity=None,
            ),
        ]

        stored_ids = await process_extracted_facts(
            facts=facts,
            store=graph_store,
            user_id="user-1",
            chat_type="private",
        )

        assert len(stored_ids) == 1
        mem = await graph_store.get_memory(stored_ids[0])
        assert mem is not None
        # PREFERENCE is not ephemeral, so no floor applied
        assert mem.sensitivity == Sensitivity.PUBLIC

    @pytest.mark.asyncio
    async def test_group_event_keeps_original_sensitivity(self, graph_store: Store):
        """Event memory in group chat keeps original sensitivity (floor only for DMs)."""
        from ash.memory.processing import process_extracted_facts
        from ash.store.types import ExtractedFact, MemoryType

        facts = [
            ExtractedFact(
                content="Team offsite next week",
                subjects=[],
                shared=False,
                confidence=0.9,
                memory_type=MemoryType.EVENT,
                sensitivity=None,
            ),
        ]

        stored_ids = await process_extracted_facts(
            facts=facts,
            store=graph_store,
            user_id="user-1",
            chat_type="group",
        )

        assert len(stored_ids) == 1
        mem = await graph_store.get_memory(stored_ids[0])
        assert mem is not None
        assert mem.sensitivity == Sensitivity.PUBLIC

    @pytest.mark.asyncio
    async def test_dm_sensitive_memory_not_downgraded(self, graph_store: Store):
        """SENSITIVE memory in DM stays SENSITIVE (floor doesn't downgrade)."""
        from ash.memory.processing import process_extracted_facts
        from ash.store.types import ExtractedFact, MemoryType

        facts = [
            ExtractedFact(
                content="Medical appointment scheduled",
                subjects=[],
                shared=False,
                confidence=0.9,
                memory_type=MemoryType.EVENT,
                sensitivity=Sensitivity.SENSITIVE,
            ),
        ]

        stored_ids = await process_extracted_facts(
            facts=facts,
            store=graph_store,
            user_id="user-1",
            chat_type="private",
        )

        assert len(stored_ids) == 1
        mem = await graph_store.get_memory(stored_ids[0])
        assert mem is not None
        assert mem.sensitivity == Sensitivity.SENSITIVE

    @pytest.mark.asyncio
    async def test_private_to_conversation_sets_chat_scoped_metadata(
        self, graph_store: Store
    ):
        """Conversation-private facts should carry chat-scoped disclosure metadata."""
        from ash.memory.processing import process_extracted_facts
        from ash.store.types import DisclosureClass, ExtractedFact, MemoryType

        facts = [
            ExtractedFact(
                content="My street address is 123 Main St",
                subjects=[],
                shared=False,
                confidence=0.9,
                memory_type=MemoryType.IDENTITY,
                sensitivity=Sensitivity.PERSONAL,
                disclosure=DisclosureClass.PRIVATE_TO_CONVERSATION,
            ),
        ]

        stored_ids = await process_extracted_facts(
            facts=facts,
            store=graph_store,
            user_id="user-1",
            chat_type="private",
        )

        assert len(stored_ids) == 1
        mem = await graph_store.get_memory(stored_ids[0])
        assert mem is not None
        assert mem.metadata is not None
        assert mem.metadata.get("conversation_private") is True
