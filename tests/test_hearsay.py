"""Tests for hearsay supersession.

Tests the logic for superseding hearsay with confirmed facts.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.store.hearsay import supersede_hearsay_for_fact
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
