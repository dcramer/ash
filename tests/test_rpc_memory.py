"""Tests for memory RPC methods.

Tests focus on:
- Input validation (API contract)
- Scoping behavior through RPC interface
"""

from collections.abc import Callable
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


@pytest.fixture
def mock_retriever():
    """Create a mock semantic retriever."""
    retriever = MagicMock()
    retriever.search_memories = AsyncMock(return_value=[])
    retriever.search = AsyncMock(return_value=[])
    retriever.index_memory = AsyncMock()
    retriever.delete_memory_embedding = AsyncMock()
    return retriever


@pytest.fixture
async def memory_manager(memory_store, mock_retriever, db_session):
    """Create a memory manager with mocked retriever."""
    return MemoryManager(
        store=memory_store,
        retriever=mock_retriever,
        db_session=db_session,
    )


@pytest.fixture
def rpc_server(memory_manager):
    """Create a mock RPC server with memory methods registered."""
    server = MockRPCServer()
    register_memory_methods(server, memory_manager)  # type: ignore[arg-type]
    return server


class TestRPCValidation:
    """Tests for RPC input validation."""

    async def test_search_requires_query(self, rpc_server):
        """Test that memory.search requires a query parameter."""
        handler = rpc_server.methods["memory.search"]

        with pytest.raises(ValueError, match="query is required"):
            await handler({})

    async def test_add_requires_content(self, rpc_server):
        """Test that memory.add requires content parameter."""
        handler = rpc_server.methods["memory.add"]

        with pytest.raises(ValueError, match="content is required"):
            await handler({})

    async def test_delete_requires_memory_id(self, rpc_server):
        """Test that memory.delete requires memory_id parameter."""
        handler = rpc_server.methods["memory.delete"]

        with pytest.raises(ValueError, match="memory_id is required"):
            await handler({})


class TestRPCScoping:
    """Tests for memory scoping through RPC interface."""

    async def test_add_creates_personal_memory_by_default(
        self, rpc_server, memory_store
    ):
        """Test that memory.add creates a personal memory by default."""
        handler = rpc_server.methods["memory.add"]

        result = await handler({"content": "Personal fact", "user_id": "user-1"})

        memory = await memory_store.get_memory(result["id"])
        assert memory.owner_user_id == "user-1"
        assert memory.chat_id is None

    async def test_add_creates_group_memory_when_shared(self, rpc_server, memory_store):
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

        memory = await memory_store.get_memory(result["id"])
        assert memory.owner_user_id is None  # Group memory has no owner
        assert memory.chat_id == "chat-1"

    async def test_add_personal_when_shared_false(self, rpc_server, memory_store):
        """Test that shared=False creates personal memory even with chat_id."""
        handler = rpc_server.methods["memory.add"]

        result = await handler(
            {
                "content": "Personal in chat",
                "user_id": "user-1",
                "chat_id": "chat-1",
                "shared": False,
            }
        )

        memory = await memory_store.get_memory(result["id"])
        assert memory.owner_user_id == "user-1"
        assert memory.chat_id is None
