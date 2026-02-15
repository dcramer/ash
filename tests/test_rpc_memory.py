"""Tests for memory RPC methods.

Tests focus on:
- Input validation (API contract)
- Scoping behavior through RPC interface
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.db.engine import Database
from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.index import VectorIndex
from ash.rpc.methods.memory import register_memory_methods
from ash.store.store import Store


class MockRPCServer:
    """Mock RPC server for testing method registration."""

    def __init__(self):
        self.methods: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, handler: Callable[..., Any]) -> None:
        self.methods[name] = handler


@pytest.fixture
def mock_embedding_generator():
    """Create a mock embedding generator."""
    generator = MagicMock(spec=EmbeddingGenerator)
    generator.embed = AsyncMock(return_value=[0.1] * 1536)
    return generator


@pytest.fixture
def mock_index():
    """Create a mock vector index."""
    index = MagicMock(spec=VectorIndex)
    index.search = AsyncMock(return_value=[])
    index.add_embedding = AsyncMock()
    index.delete_embedding = AsyncMock()
    index.delete_embeddings = AsyncMock()
    return index


@pytest.fixture
async def memory_manager(
    database: Database, mock_index, mock_embedding_generator
) -> Store:
    """Create a Store with mocked components."""
    return Store(
        db=database,
        vector_index=mock_index,
        embedding_generator=mock_embedding_generator,
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


class TestRPCForgetPerson:
    """Tests for memory.forget_person RPC method."""

    async def test_forget_person_requires_person_id(self, rpc_server):
        """Test that memory.forget_person requires person_id parameter."""
        handler = rpc_server.methods["memory.forget_person"]

        with pytest.raises(ValueError, match="person_id is required"):
            await handler({})

    async def test_forget_person_archives_memories(self, rpc_server, memory_manager):
        """Test that memory.forget_person archives subject memories."""
        person = await memory_manager.create_person(
            created_by="alice", name="Bob", aliases=["bob"]
        )

        await memory_manager.add_memory(
            content="Bob likes hiking",
            owner_user_id="alice",
            subject_person_ids=[person.id],
        )
        await memory_manager.add_memory(
            content="Alice likes cooking",
            owner_user_id="alice",
        )

        handler = rpc_server.methods["memory.forget_person"]
        result = await handler({"person_id": person.id})

        assert result["archived_count"] == 1

        # Verify only Bob's memory was removed
        remaining = await memory_manager.list_memories()
        assert len(remaining) == 1
        assert remaining[0].content == "Alice likes cooking"


class TestRPCScoping:
    """Tests for memory scoping through RPC interface."""

    async def test_add_creates_personal_memory_by_default(
        self, rpc_server, memory_manager
    ):
        """Test that memory.add creates a personal memory by default."""
        handler = rpc_server.methods["memory.add"]

        result = await handler({"content": "Personal fact", "user_id": "user-1"})

        memory = await memory_manager.get_memory(result["id"])
        assert memory is not None
        assert memory.owner_user_id == "user-1"
        assert memory.chat_id is None

    async def test_add_creates_group_memory_when_shared(
        self, rpc_server, memory_manager
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

        memory = await memory_manager.get_memory(result["id"])
        assert memory is not None
        assert memory.owner_user_id is None  # Group memory has no owner
        assert memory.chat_id == "chat-1"

    async def test_add_personal_when_shared_false(self, rpc_server, memory_manager):
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

        memory = await memory_manager.get_memory(result["id"])
        assert memory is not None
        assert memory.owner_user_id == "user-1"
        assert memory.chat_id is None
