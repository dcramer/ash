"""Tests for HearsayManager and hearsay supersession.

Tests the logic for superseding hearsay with confirmed facts.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.store.hearsay import HearsayManager, supersede_hearsay_for_fact
from ash.store.types import MemoryEntry, MemoryType


def make_memory(
    *,
    id: str = "mem-1",
    content: str = "Test memory",
    subject_person_ids: list[str] | None = None,
    source_username: str | None = "testuser",
) -> MemoryEntry:
    """Create a MemoryEntry for testing."""
    return MemoryEntry(
        id=id,
        version=1,
        content=content,
        memory_type=MemoryType.KNOWLEDGE,
        owner_user_id="user-1",
        source_username=source_username,
        subject_person_ids=subject_person_ids or [],
        created_at=datetime.now(UTC),
    )


class TestHearsayManager:
    """Tests for HearsayManager class."""

    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        store.supersede_confirmed_hearsay = AsyncMock(return_value=0)
        return store

    @pytest.mark.asyncio
    async def test_handle_confirmed_fact_skips_non_self_facts(self, mock_store):
        """Facts about others (with subject_person_ids) should be skipped."""
        manager = HearsayManager(mock_store)

        # Memory about someone else (has subject_person_ids)
        memory = make_memory(subject_person_ids=["person-1"])

        result = await manager.handle_confirmed_fact(
            new_memory=memory,
            speaker_person_ids={"person-2"},
            source_username="testuser",
            owner_user_id="user-1",
        )

        # Should not call supersede since this is not a self-fact
        mock_store.supersede_confirmed_hearsay.assert_not_called()
        assert result.superseded_count == 0

    @pytest.mark.asyncio
    async def test_handle_confirmed_fact_skips_without_speaker_ids(self, mock_store):
        """Self-facts without speaker person IDs should be skipped."""
        manager = HearsayManager(mock_store)

        # Self-fact (no subject_person_ids)
        memory = make_memory(subject_person_ids=[])

        result = await manager.handle_confirmed_fact(
            new_memory=memory,
            speaker_person_ids=set(),  # Empty speaker IDs
            source_username="testuser",
            owner_user_id="user-1",
        )

        mock_store.supersede_confirmed_hearsay.assert_not_called()
        assert result.superseded_count == 0

    @pytest.mark.asyncio
    async def test_handle_confirmed_fact_calls_supersede(self, mock_store):
        """Valid self-facts should trigger hearsay supersession."""
        mock_store.supersede_confirmed_hearsay = AsyncMock(return_value=2)
        manager = HearsayManager(mock_store)

        # Self-fact (no subject_person_ids = speaker talking about themselves)
        memory = make_memory(subject_person_ids=[])

        result = await manager.handle_confirmed_fact(
            new_memory=memory,
            speaker_person_ids={"person-1"},
            source_username="testuser",
            owner_user_id="user-1",
        )

        mock_store.supersede_confirmed_hearsay.assert_called_once_with(
            new_memory=memory,
            person_ids={"person-1"},
            source_username="testuser",
            owner_user_id="user-1",
        )
        assert result.superseded_count == 2

    @pytest.mark.asyncio
    async def test_handle_confirmed_fact_handles_exception(self, mock_store):
        """Exceptions should be caught and return zero."""
        mock_store.supersede_confirmed_hearsay = AsyncMock(
            side_effect=Exception("DB error")
        )
        manager = HearsayManager(mock_store)

        memory = make_memory(subject_person_ids=[])

        result = await manager.handle_confirmed_fact(
            new_memory=memory,
            speaker_person_ids={"person-1"},
            source_username="testuser",
            owner_user_id="user-1",
        )

        assert result.superseded_count == 0
        assert result.checked_count == 0


class TestSupersedHearsayForFact:
    """Tests for supersede_hearsay_for_fact convenience function."""

    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        store.find_person_ids_for_username = AsyncMock(return_value=set())
        store.supersede_confirmed_hearsay = AsyncMock(return_value=0)
        return store

    @pytest.mark.asyncio
    async def test_skips_facts_about_others(self, mock_store):
        """Facts with subject_person_ids (about others) should be skipped."""
        memory = make_memory(subject_person_ids=["person-1"])

        result = await supersede_hearsay_for_fact(
            store=mock_store,
            new_memory=memory,
            source_username="testuser",
            owner_user_id="user-1",
        )

        assert result == 0
        mock_store.find_person_ids_for_username.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_without_source_username(self, mock_store):
        """Facts without source_username should be skipped."""
        memory = make_memory(subject_person_ids=[], source_username=None)

        result = await supersede_hearsay_for_fact(
            store=mock_store,
            new_memory=memory,
            source_username=None,
            owner_user_id="user-1",
        )

        assert result == 0
        mock_store.find_person_ids_for_username.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_if_no_person_ids_found(self, mock_store):
        """Should skip if source username doesn't resolve to person IDs."""
        mock_store.find_person_ids_for_username = AsyncMock(return_value=set())
        memory = make_memory(subject_person_ids=[])

        result = await supersede_hearsay_for_fact(
            store=mock_store,
            new_memory=memory,
            source_username="testuser",
            owner_user_id="user-1",
        )

        assert result == 0
        mock_store.supersede_confirmed_hearsay.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_supersede_with_resolved_person_ids(self, mock_store):
        """Should call supersede with resolved person IDs."""
        mock_store.find_person_ids_for_username = AsyncMock(
            return_value={"person-1", "person-2"}
        )
        mock_store.supersede_confirmed_hearsay = AsyncMock(return_value=3)

        memory = make_memory(subject_person_ids=[])

        result = await supersede_hearsay_for_fact(
            store=mock_store,
            new_memory=memory,
            source_username="testuser",
            owner_user_id="user-1",
        )

        assert result == 3
        mock_store.find_person_ids_for_username.assert_called_once_with("testuser")
        mock_store.supersede_confirmed_hearsay.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self, mock_store):
        """Exceptions should be caught and return zero."""
        mock_store.find_person_ids_for_username = AsyncMock(
            side_effect=Exception("DB error")
        )

        memory = make_memory(subject_person_ids=[])

        result = await supersede_hearsay_for_fact(
            store=mock_store,
            new_memory=memory,
            source_username="testuser",
            owner_user_id="user-1",
        )

        assert result == 0
