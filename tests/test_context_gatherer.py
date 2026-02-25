"""Tests for ContextGatherer.

Tests the context gathering logic extracted from Agent.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.core.context import ContextGatherer, GatheredContext
from ash.memory.query_planner import PlannedMemoryQuery
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

    @pytest.mark.asyncio
    async def test_planner_rewrites_to_single_retrieval_query(self):
        """Planner should rewrite to one retrieval query with planner fetch limit."""
        result_context = RetrievedContext(
            memories=[
                SearchResult(
                    id="mem-location",
                    content="You live in San Francisco",
                    similarity=0.95,
                    metadata={},
                    source_type="memory",
                )
            ]
        )
        planner = MagicMock()
        planner.plan = AsyncMock(
            return_value=PlannedMemoryQuery(query="my location city", max_results=25)
        )

        mock_store = MagicMock()
        mock_store.get_context_for_message = AsyncMock(return_value=result_context)
        mock_store.list_people = AsyncMock(return_value=[])
        mock_store.find_person_ids_for_username = AsyncMock(return_value=set())
        mock_store.get_person = AsyncMock(return_value=None)

        gatherer = ContextGatherer(
            store=mock_store,
            query_planner=planner,
            max_total_memories=10,
        )
        result = await gatherer.gather(user_id="user-1", user_message="check weather")

        assert result.memory is not None
        assert [m.id for m in result.memory.memories] == ["mem-location"]
        assert mock_store.get_context_for_message.await_count == 1
        first_call = mock_store.get_context_for_message.await_args_list[0]
        assert first_call.kwargs["user_message"] == "my location city"
        assert first_call.kwargs["max_memories"] == 25
        planner.plan.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_memory_retrieval_log_includes_query_and_ids(self, caplog):
        """Retrieval log should include resolved query text and memory IDs."""
        planner = MagicMock()
        planner.plan = AsyncMock(
            return_value=PlannedMemoryQuery(
                query="planned lookup query",
                max_results=25,
                supplemental_queries=("user city",),
            )
        )
        result_context = RetrievedContext(
            memories=[
                SearchResult(
                    id="mem-123",
                    content="User lives in SF",
                    similarity=0.95,
                    metadata={},
                    source_type="memory",
                )
            ]
        )
        mock_store = MagicMock()
        mock_store.get_context_for_message = AsyncMock(return_value=result_context)
        mock_store.list_people = AsyncMock(return_value=[])
        mock_store.find_person_ids_for_username = AsyncMock(return_value=set())
        mock_store.get_person = AsyncMock(return_value=None)

        gatherer = ContextGatherer(store=mock_store, query_planner=planner)

        with caplog.at_level("INFO", logger="ash.core.context"):
            await gatherer.gather(user_id="user-1", user_message="weather?")

        record = next((r for r in caplog.records if r.msg == "memory_retrieval"), None)
        assert record is not None
        assert record.__dict__["memory.query"] == "planned lookup query"
        assert record.__dict__["memory.lookup_queries"] == ["user city"]
        assert record.__dict__["memory.ids"] == ["mem-123"]

    @pytest.mark.asyncio
    async def test_planner_receives_recent_chat_context(self, monkeypatch):
        """Planner should receive recent chat messages for better lookup planning."""
        planner = MagicMock()
        planner.plan = AsyncMock(
            return_value=PlannedMemoryQuery(query="weather", max_results=25)
        )
        mock_store = MagicMock()
        mock_store.get_context_for_message = AsyncMock(
            return_value=RetrievedContext(memories=[])
        )
        mock_store.list_people = AsyncMock(return_value=[])
        mock_store.find_person_ids_for_username = AsyncMock(return_value=set())
        mock_store.get_person = AsyncMock(return_value=None)

        from datetime import UTC, datetime

        from ash.chats.history import HistoryEntry

        def _fake_recent(*_args, **_kwargs):
            return [
                HistoryEntry(
                    id="1",
                    role="user",
                    content="we're traveling this week",
                    created_at=datetime.now(UTC),
                    username="alice",
                ),
                HistoryEntry(
                    id="2",
                    role="assistant",
                    content="noted, where are you based?",
                    created_at=datetime.now(UTC),
                ),
            ]

        monkeypatch.setattr(
            "ash.chats.history.read_recent_chat_history",
            _fake_recent,
        )

        gatherer = ContextGatherer(store=mock_store, query_planner=planner)
        await gatherer.gather(
            user_id="user-1",
            user_message="what's the weather",
            provider="telegram",
            chat_id="chat-1",
            sender_username="alice",
        )

        planner.plan.assert_awaited_once()
        kwargs = planner.plan.await_args.kwargs
        assert kwargs["recent_messages"] == (
            "user:alice: we're traveling this week",
            "assistant: noted, where are you based?",
        )

    @pytest.mark.asyncio
    async def test_planner_failure_falls_back_to_base_query(self):
        """Planner failure should not block normal retrieval."""
        planner = MagicMock()
        planner.plan = AsyncMock(side_effect=RuntimeError("timeout"))
        mock_store = MagicMock()
        mock_store.get_context_for_message = AsyncMock(
            return_value=RetrievedContext(memories=[])
        )
        mock_store.list_people = AsyncMock(return_value=[])
        mock_store.find_person_ids_for_username = AsyncMock(return_value=set())
        mock_store.get_person = AsyncMock(return_value=None)

        gatherer = ContextGatherer(store=mock_store, query_planner=planner)
        await gatherer.gather(
            user_id="user-1", user_message="what did I say yesterday?"
        )

        assert mock_store.get_context_for_message.await_count == 1

    @pytest.mark.asyncio
    async def test_planner_lookup_queries_merge_and_dedup_results(self):
        """Planner supplemental queries should be retrieved and merged."""
        planner = MagicMock()
        planner.plan = AsyncMock(
            return_value=PlannedMemoryQuery(
                query="weather",
                max_results=25,
                supplemental_queries=("where user lives", "user city"),
            )
        )
        mock_store = MagicMock()
        mock_store.get_context_for_message = AsyncMock(
            side_effect=[
                RetrievedContext(
                    memories=[
                        SearchResult(
                            id="mem-weather",
                            content="General weather preference",
                            similarity=0.4,
                            metadata={},
                            source_type="memory",
                        ),
                        SearchResult(
                            id="mem-location",
                            content="User lives in San Francisco",
                            similarity=0.7,
                            metadata={},
                            source_type="memory",
                        ),
                    ]
                ),
                RetrievedContext(
                    memories=[
                        SearchResult(
                            id="mem-location",
                            content="User lives in San Francisco",
                            similarity=0.95,
                            metadata={},
                            source_type="memory",
                        )
                    ]
                ),
                RetrievedContext(memories=[]),
            ]
        )
        mock_store.list_people = AsyncMock(return_value=[])
        mock_store.find_person_ids_for_username = AsyncMock(return_value=set())
        mock_store.get_person = AsyncMock(return_value=None)

        gatherer = ContextGatherer(store=mock_store, query_planner=planner)
        result = await gatherer.gather(user_id="user-1", user_message="weather?")

        assert result.memory is not None
        assert [m.id for m in result.memory.memories] == [
            "mem-location",
            "mem-weather",
        ]
        first = mock_store.get_context_for_message.await_args_list[0]
        second = mock_store.get_context_for_message.await_args_list[1]
        third = mock_store.get_context_for_message.await_args_list[2]
        assert first.kwargs["user_message"] == "weather"
        assert second.kwargs["user_message"] == "where user lives"
        assert third.kwargs["user_message"] == "user city"

    @pytest.mark.asyncio
    async def test_planner_lookup_query_failure_is_partial(self):
        """A failing supplemental retrieval should not discard successful ones."""
        planner = MagicMock()
        planner.plan = AsyncMock(
            return_value=PlannedMemoryQuery(
                query="weather",
                max_results=25,
                supplemental_queries=("where user lives",),
            )
        )
        mock_store = MagicMock()
        mock_store.get_context_for_message = AsyncMock(
            side_effect=[
                RetrievedContext(
                    memories=[
                        SearchResult(
                            id="mem-location",
                            content="User lives in San Francisco",
                            similarity=0.9,
                            metadata={},
                            source_type="memory",
                        )
                    ]
                ),
                RuntimeError("transient retrieval failure"),
            ]
        )
        mock_store.list_people = AsyncMock(return_value=[])
        mock_store.find_person_ids_for_username = AsyncMock(return_value=set())
        mock_store.get_person = AsyncMock(return_value=None)

        gatherer = ContextGatherer(store=mock_store, query_planner=planner)
        result = await gatherer.gather(user_id="user-1", user_message="weather?")

        assert result.memory is not None
        assert [m.id for m in result.memory.memories] == ["mem-location"]
