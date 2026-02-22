"""Tests for ContextGatherer.

Tests the context gathering logic extracted from Agent.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.core.context import ContextGatherer, GatheredContext
from ash.store.types import PersonEntry, RetrievedContext, SearchResult


class TestContextGatherer:
    """Tests for ContextGatherer."""

    @pytest.mark.asyncio
    async def test_gather_without_store_returns_empty_context(self):
        """When store is None, gather returns empty GatheredContext."""
        gatherer = ContextGatherer(store=None)

        result = await gatherer.gather(user_id="user-1", user_message="hello")

        assert result == GatheredContext()
        assert result.memory is None
        assert result.known_people is None

    @pytest.mark.asyncio
    async def test_gather_without_user_id_returns_empty_context(self):
        """When user_id is None, gather returns empty context."""
        mock_store = MagicMock()
        gatherer = ContextGatherer(store=mock_store)

        result = await gatherer.gather(user_id=None, user_message="hello")

        assert result.memory is None
        assert result.known_people is None

    @pytest.mark.asyncio
    async def test_gather_calls_store_methods(self):
        """Gather should call store.get_context_for_message and list_people."""
        mock_store = MagicMock()
        mock_store.get_context_for_message = AsyncMock(
            return_value=RetrievedContext(memories=[])
        )
        mock_store.list_people = AsyncMock(return_value=[])
        mock_store.find_person_ids_for_username = AsyncMock(return_value=set())
        mock_store.get_person = AsyncMock(return_value=None)

        gatherer = ContextGatherer(store=mock_store)

        result = await gatherer.gather(
            user_id="user-1",
            user_message="What do you know about Sarah?",
            chat_id="chat-1",
            chat_type="private",
            sender_username="johndoe",
        )

        mock_store.get_context_for_message.assert_called_once()
        mock_store.list_people.assert_called_once()
        assert result.memory is not None
        assert result.known_people is not None

    @pytest.mark.asyncio
    async def test_gather_resolves_sender_person_ids(self):
        """Gather should resolve sender username to person IDs for cross-context."""
        mock_store = MagicMock()
        mock_store.find_person_ids_for_username = AsyncMock(
            return_value={"person-1", "person-2"}
        )
        mock_store.get_person = AsyncMock(
            side_effect=[
                PersonEntry(
                    id="person-1",
                    version=1,
                    created_by="user-1",
                    name="John Doe",
                    aliases=[],
                    relationships=[],
                ),
                PersonEntry(
                    id="person-2",
                    version=1,
                    created_by="user-1",
                    name="John D",
                    aliases=[],
                    relationships=[],
                ),
            ]
        )
        mock_store.get_context_for_message = AsyncMock(
            return_value=RetrievedContext(memories=[])
        )
        mock_store.list_people = AsyncMock(return_value=[])

        gatherer = ContextGatherer(store=mock_store)

        await gatherer.gather(
            user_id="user-1",
            user_message="hello",
            sender_username="johndoe",
        )

        mock_store.find_person_ids_for_username.assert_called_once_with("johndoe")
        # Verify participant_person_ids was passed to get_context_for_message
        call_kwargs = mock_store.get_context_for_message.call_args.kwargs
        assert call_kwargs["participant_person_ids"] == {
            "johndoe": {"person-1", "person-2"}
        }
        assert mock_store.get_person.call_count == 2

    @pytest.mark.asyncio
    async def test_gather_handles_memory_retrieval_failure(self):
        """Gather should handle exceptions in memory retrieval gracefully."""
        mock_store = MagicMock()
        mock_store.get_context_for_message = AsyncMock(
            side_effect=Exception("Database error")
        )
        mock_store.list_people = AsyncMock(return_value=[])
        mock_store.find_person_ids_for_username = AsyncMock(return_value=set())
        mock_store.get_person = AsyncMock(return_value=None)

        gatherer = ContextGatherer(store=mock_store)

        result = await gatherer.gather(
            user_id="user-1",
            user_message="hello",
        )

        # Should return None for memory, not raise
        assert result.memory is None
        # Known people should still be populated
        assert result.known_people == []

    @pytest.mark.asyncio
    async def test_gather_handles_list_people_failure(self):
        """Gather should handle exceptions in list_people gracefully."""
        mock_store = MagicMock()
        mock_store.get_context_for_message = AsyncMock(
            return_value=RetrievedContext(memories=[])
        )
        mock_store.list_people = AsyncMock(side_effect=Exception("Database error"))
        mock_store.find_person_ids_for_username = AsyncMock(return_value=set())
        mock_store.get_person = AsyncMock(return_value=None)

        gatherer = ContextGatherer(store=mock_store)

        result = await gatherer.gather(
            user_id="user-1",
            user_message="hello",
        )

        # Memory should still be populated
        assert result.memory is not None
        # Known people should be None due to failure
        assert result.known_people is None

    @pytest.mark.asyncio
    async def test_gather_returns_memories_and_people(self):
        """Gather should return both memories and people when available."""
        mock_memory = SearchResult(
            id="mem-1",
            content="User likes pizza",
            similarity=0.9,
            metadata={},
            source_type="memory",
        )
        mock_person = PersonEntry(
            id="person-1",
            version=1,
            created_by="user-1",
            name="Sarah",
            aliases=[],
            relationships=[],
        )

        mock_store = MagicMock()
        mock_store.get_context_for_message = AsyncMock(
            return_value=RetrievedContext(memories=[mock_memory])
        )
        mock_store.list_people = AsyncMock(return_value=[mock_person])
        mock_store.find_person_ids_for_username = AsyncMock(return_value=set())
        mock_store.get_person = AsyncMock(return_value=None)

        gatherer = ContextGatherer(store=mock_store)

        result = await gatherer.gather(
            user_id="user-1",
            user_message="hello",
        )

        assert result.memory is not None
        assert len(result.memory.memories) == 1
        assert result.memory.memories[0].content == "User likes pizza"
        assert result.known_people is not None
        assert len(result.known_people) == 1
        assert result.known_people[0].name == "Sarah"
