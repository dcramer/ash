"""Tests for memory RPC methods."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.memory import MemoryManager
from ash.rpc.methods.memory import register_memory_methods


class MockRPCServer:
    """Mock RPC server for testing method registration."""

    def __init__(self):
        self.methods: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, handler: Callable[..., Any]) -> None:
        self.methods[name] = handler


class TestMemoryRPCMethods:
    """Tests for memory RPC method handlers."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock semantic retriever."""
        retriever = MagicMock()
        retriever.search_memories = AsyncMock(return_value=[])
        retriever.search = AsyncMock(return_value=[])
        retriever.index_memory = AsyncMock()
        retriever.delete_memory_embedding = AsyncMock()
        return retriever

    @pytest.fixture
    async def memory_manager(self, memory_store, mock_retriever, db_session):
        """Create a memory manager with mocked retriever."""
        return MemoryManager(
            store=memory_store,
            retriever=mock_retriever,
            db_session=db_session,
        )

    @pytest.fixture
    def rpc_server(self, memory_manager):
        """Create a mock RPC server with memory methods registered."""
        server = MockRPCServer()
        register_memory_methods(server, memory_manager)  # type: ignore[arg-type]
        return server

    # memory.search tests

    async def test_memory_search_requires_query(self, rpc_server):
        """Test that memory.search requires a query parameter."""
        handler = rpc_server.methods["memory.search"]

        with pytest.raises(ValueError, match="query is required"):
            await handler({})

    async def test_memory_search_returns_results(
        self, rpc_server, memory_manager, mock_retriever
    ):
        """Test that memory.search returns search results."""
        from ash.memory import SearchResult

        mock_retriever.search.return_value = [
            SearchResult(
                id="mem-1",
                content="Test fact",
                similarity=0.9,
                source_type="memory",
                metadata={"key": "value"},
            )
        ]

        handler = rpc_server.methods["memory.search"]
        results = await handler({"query": "test", "limit": 5})

        assert len(results) == 1
        assert results[0]["id"] == "mem-1"
        assert results[0]["content"] == "Test fact"
        assert results[0]["similarity"] == 0.9
        assert results[0]["metadata"] == {"key": "value"}

    async def test_memory_search_passes_scoping_params(
        self, rpc_server, memory_manager, mock_retriever
    ):
        """Test that memory.search passes user_id and chat_id to manager."""
        handler = rpc_server.methods["memory.search"]
        await handler(
            {
                "query": "test",
                "user_id": "user-1",
                "chat_id": "chat-1",
            }
        )

        mock_retriever.search.assert_called_once()
        call_kwargs = mock_retriever.search.call_args.kwargs
        assert call_kwargs["owner_user_id"] == "user-1"
        assert call_kwargs["chat_id"] == "chat-1"

    # memory.add tests

    async def test_memory_add_requires_content(self, rpc_server):
        """Test that memory.add requires content parameter."""
        handler = rpc_server.methods["memory.add"]

        with pytest.raises(ValueError, match="content is required"):
            await handler({})

    async def test_memory_add_creates_personal_memory(self, rpc_server, memory_store):
        """Test that memory.add creates a personal memory by default."""
        handler = rpc_server.methods["memory.add"]

        result = await handler(
            {
                "content": "Personal fact",
                "user_id": "user-1",
            }
        )

        assert "id" in result

        # Verify the memory was created with correct scoping
        memory = await memory_store.get_memory(result["id"])
        assert memory.content == "Personal fact"
        assert memory.owner_user_id == "user-1"
        assert memory.chat_id is None

    async def test_memory_add_creates_group_memory_with_shared_flag(
        self, rpc_server, memory_store
    ):
        """Test that memory.add creates a group memory when shared=True."""
        handler = rpc_server.methods["memory.add"]

        result = await handler(
            {
                "content": "Group fact",
                "user_id": "user-1",
                "chat_id": "chat-1",
                "shared": True,
            }
        )

        # Verify the memory was created with correct scoping
        memory = await memory_store.get_memory(result["id"])
        assert memory.content == "Group fact"
        assert memory.owner_user_id is None  # Group memory has no owner
        assert memory.chat_id == "chat-1"

    async def test_memory_add_personal_when_shared_false(
        self, rpc_server, memory_store
    ):
        """Test that memory.add creates personal memory when shared=False even with chat_id."""
        handler = rpc_server.methods["memory.add"]

        result = await handler(
            {
                "content": "Personal in chat",
                "user_id": "user-1",
                "chat_id": "chat-1",
                "shared": False,
            }
        )

        # Should be personal, not group
        memory = await memory_store.get_memory(result["id"])
        assert memory.owner_user_id == "user-1"
        assert memory.chat_id is None  # Chat ID not used for personal memories

    async def test_memory_add_with_expiration(self, rpc_server, memory_store):
        """Test that memory.add handles expires_days parameter."""
        handler = rpc_server.methods["memory.add"]

        result = await handler(
            {
                "content": "Temporary fact",
                "expires_days": 7,
            }
        )

        memory = await memory_store.get_memory(result["id"])
        assert memory.expires_at is not None
        # expires_at should be in the future (roughly 7 days from now)
        # Handle both naive and aware datetimes
        now = datetime.now(UTC)
        expires_at = memory.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        assert expires_at > now

    async def test_memory_add_with_source(self, rpc_server, memory_store):
        """Test that memory.add uses custom source label."""
        handler = rpc_server.methods["memory.add"]

        result = await handler(
            {
                "content": "From tool",
                "source": "custom_tool",
            }
        )

        memory = await memory_store.get_memory(result["id"])
        assert memory.source == "custom_tool"

    # memory.list tests

    async def test_memory_list_returns_memories(self, rpc_server, memory_store):
        """Test that memory.list returns recent memories."""
        # Add some memories
        await memory_store.add_memory(content="Fact 1", source="test")
        await memory_store.add_memory(content="Fact 2", source="test")

        handler = rpc_server.methods["memory.list"]
        results = await handler({"limit": 10})

        assert len(results) == 2
        assert all("id" in m for m in results)
        assert all("content" in m for m in results)
        assert all("source" in m for m in results)

    async def test_memory_list_filters_by_user(self, rpc_server, memory_store):
        """Test that memory.list filters by user_id."""
        await memory_store.add_memory(
            content="User 1 fact",
            owner_user_id="user-1",
        )
        await memory_store.add_memory(
            content="User 2 fact",
            owner_user_id="user-2",
        )

        handler = rpc_server.methods["memory.list"]
        results = await handler({"user_id": "user-1"})

        assert len(results) == 1
        assert results[0]["content"] == "User 1 fact"

    async def test_memory_list_filters_by_chat(self, rpc_server, memory_store):
        """Test that memory.list filters by chat_id for group memories."""
        await memory_store.add_memory(
            content="Chat 1 fact",
            owner_user_id=None,
            chat_id="chat-1",
        )
        await memory_store.add_memory(
            content="Chat 2 fact",
            owner_user_id=None,
            chat_id="chat-2",
        )

        handler = rpc_server.methods["memory.list"]
        results = await handler({"chat_id": "chat-1"})

        assert len(results) == 1
        assert results[0]["content"] == "Chat 1 fact"

    async def test_memory_list_excludes_expired_by_default(
        self, rpc_server, memory_store
    ):
        """Test that memory.list excludes expired memories by default."""
        past = datetime.now(UTC) - timedelta(days=1)
        await memory_store.add_memory(
            content="Expired fact",
            expires_at=past,
        )
        await memory_store.add_memory(content="Valid fact")

        handler = rpc_server.methods["memory.list"]
        results = await handler({})

        assert len(results) == 1
        assert results[0]["content"] == "Valid fact"

    async def test_memory_list_includes_expired_when_requested(
        self, rpc_server, memory_store
    ):
        """Test that memory.list can include expired memories."""
        past = datetime.now(UTC) - timedelta(days=1)
        await memory_store.add_memory(
            content="Expired fact",
            expires_at=past,
        )
        await memory_store.add_memory(content="Valid fact")

        handler = rpc_server.methods["memory.list"]
        results = await handler({"include_expired": True})

        assert len(results) == 2

    # memory.delete tests

    async def test_memory_delete_requires_memory_id(self, rpc_server):
        """Test that memory.delete requires memory_id parameter."""
        handler = rpc_server.methods["memory.delete"]

        with pytest.raises(ValueError, match="memory_id is required"):
            await handler({})

    async def test_memory_delete_removes_memory(self, rpc_server, memory_store):
        """Test that memory.delete removes a memory."""
        memory = await memory_store.add_memory(content="To be deleted")

        handler = rpc_server.methods["memory.delete"]
        result = await handler({"memory_id": memory.id})

        assert result["deleted"] is True

        # Verify memory is gone
        deleted = await memory_store.get_memory(memory.id)
        assert deleted is None

    async def test_memory_delete_returns_false_for_nonexistent(self, rpc_server):
        """Test that memory.delete returns false for nonexistent memory."""
        handler = rpc_server.methods["memory.delete"]
        result = await handler({"memory_id": "nonexistent-id"})

        assert result["deleted"] is False


class TestRPCMethodRegistration:
    """Tests for RPC method registration."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock semantic retriever."""
        retriever = MagicMock()
        retriever.search_memories = AsyncMock(return_value=[])
        retriever.search = AsyncMock(return_value=[])
        retriever.index_memory = AsyncMock()
        retriever.delete_memory_embedding = AsyncMock()
        return retriever

    @pytest.fixture
    async def memory_manager(self, memory_store, mock_retriever, db_session):
        """Create a memory manager with mocked retriever."""
        return MemoryManager(
            store=memory_store,
            retriever=mock_retriever,
            db_session=db_session,
        )

    def test_all_methods_registered(self, memory_manager):
        """Test that all expected methods are registered."""
        server = MockRPCServer()
        register_memory_methods(server, memory_manager)  # type: ignore[arg-type]

        assert "memory.search" in server.methods
        assert "memory.add" in server.methods
        assert "memory.list" in server.methods
        assert "memory.delete" in server.methods
