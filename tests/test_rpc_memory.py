"""Tests for memory RPC methods.

Tests focus on:
- Input validation (API contract)
- Scoping behavior through RPC interface
- Enriched memory_add with classification
- memory.extract handler
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.graph.graph import KnowledgeGraph
from ash.graph.persistence import GraphPersistence
from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.extractor import MemoryExtractor
from ash.rpc.methods.memory import register_memory_methods
from ash.store.store import Store
from ash.store.types import ExtractedFact, MemoryType, Sensitivity


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
    index = MagicMock()
    index.search = AsyncMock(return_value=[])
    index.add = AsyncMock()
    index.remove = AsyncMock()
    return index


@pytest.fixture
async def memory_manager(graph_dir, mock_index, mock_embedding_generator) -> Store:
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


@pytest.fixture
def mock_extractor():
    """Create a mock MemoryExtractor."""
    extractor = MagicMock(spec=MemoryExtractor)
    extractor.classify_fact = AsyncMock(return_value=None)
    extractor.extract_from_conversation = AsyncMock(return_value=[])
    return extractor


@pytest.fixture
def rpc_server(memory_manager, mock_extractor, tmp_path):
    """Create a mock RPC server with memory methods registered."""
    server = MockRPCServer()
    sessions_path = tmp_path / "sessions"
    sessions_path.mkdir()
    register_memory_methods(
        server,  # type: ignore[arg-type]
        memory_manager,
        person_manager=memory_manager,
        memory_extractor=mock_extractor,
        sessions_path=sessions_path,
    )
    return server


@pytest.fixture
def rpc_server_no_extractor(memory_manager):
    """Create a mock RPC server without memory extractor."""
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


class TestRPCMemoryAdd:
    """Tests for enriched memory_add handler."""

    async def test_add_with_classification(
        self, rpc_server, memory_manager, mock_extractor
    ):
        """Test that classify_fact enriches subjects/type when no explicit subjects."""
        mock_extractor.classify_fact.return_value = ExtractedFact(
            content="Sarah is my sister",
            subjects=["Sarah"],
            shared=False,
            confidence=1.0,
            memory_type=MemoryType.RELATIONSHIP,
            sensitivity=Sensitivity.PUBLIC,
            portable=True,
        )

        handler = rpc_server.methods["memory.add"]
        result = await handler(
            {
                "content": "Sarah is my sister",
                "user_id": "user-1",
                "source_username": "david",
                "source_display_name": "David Cramer",
            }
        )

        assert "id" in result
        mock_extractor.classify_fact.assert_awaited_once_with("Sarah is my sister")

        # Verify the memory was stored with the classified type
        memory = await memory_manager.get_memory(result["id"])
        assert memory is not None
        assert memory.memory_type == MemoryType.RELATIONSHIP

    async def test_add_explicit_subjects_skips_classification(
        self, rpc_server, memory_manager, mock_extractor
    ):
        """Test that explicit subjects parameter skips LLM classification."""
        handler = rpc_server.methods["memory.add"]
        result = await handler(
            {
                "content": "Sarah is my sister",
                "user_id": "user-1",
                "subjects": ["Sarah"],
            }
        )

        assert "id" in result
        mock_extractor.classify_fact.assert_not_awaited()

    async def test_add_without_extractor_falls_back(
        self, rpc_server_no_extractor, memory_manager
    ):
        """Test graceful memory add without extractor available."""
        handler = rpc_server_no_extractor.methods["memory.add"]
        result = await handler(
            {
                "content": "A simple fact",
                "user_id": "user-1",
            }
        )

        assert "id" in result
        memory = await memory_manager.get_memory(result["id"])
        assert memory is not None
        assert memory.content == "A simple fact"

    async def test_add_classification_failure_still_stores(
        self, rpc_server, memory_manager, mock_extractor
    ):
        """Test that classification failure doesn't prevent storage."""
        mock_extractor.classify_fact.return_value = None

        handler = rpc_server.methods["memory.add"]
        result = await handler(
            {
                "content": "Some fact to store",
                "user_id": "user-1",
            }
        )

        assert "id" in result
        memory = await memory_manager.get_memory(result["id"])
        assert memory is not None
        assert memory.content == "Some fact to store"


class TestRPCMemoryExtract:
    """Tests for memory.extract handler."""

    async def test_extract_requires_extractor(self, rpc_server_no_extractor):
        """Test that memory.extract requires extractor to be available."""
        handler = rpc_server_no_extractor.methods["memory.extract"]

        with pytest.raises(ValueError, match="Memory extractor not available"):
            await handler(
                {
                    "message_id": "msg-1",
                    "provider": "telegram",
                    "user_id": "user-1",
                }
            )

    async def test_extract_requires_message_id(self, rpc_server):
        """Test that memory.extract requires message_id."""
        handler = rpc_server.methods["memory.extract"]

        with pytest.raises(ValueError, match="message_id is required"):
            await handler({"provider": "telegram"})

    async def test_extract_requires_provider(self, rpc_server):
        """Test that memory.extract requires provider."""
        handler = rpc_server.methods["memory.extract"]

        with pytest.raises(ValueError, match="provider is required"):
            await handler({"message_id": "msg-1"})

    async def test_extract_returns_zero_when_message_not_found(
        self, rpc_server, tmp_path
    ):
        """Test that extract returns 0 when message not found in session."""
        handler = rpc_server.methods["memory.extract"]

        result = await handler(
            {
                "message_id": "nonexistent-msg",
                "provider": "telegram",
                "user_id": "user-1",
                "chat_id": "chat-1",
            }
        )

        assert result["stored"] == 0

    async def test_extract_runs_full_pipeline(
        self, memory_manager, mock_extractor, tmp_path
    ):
        """Test that extract reads session, runs extraction, and processes facts."""
        import json
        from datetime import UTC, datetime

        from ash.sessions.types import session_key

        # Set up session directory with a message
        key = session_key("telegram", "chat-1", "user-1")
        session_dir = tmp_path / "sessions" / key
        session_dir.mkdir(parents=True)

        msg_id = "test-msg-123"
        context_file = session_dir / "context.jsonl"
        entries = [
            {
                "type": "session",
                "id": "session-1",
                "created_at": datetime.now(UTC).isoformat(),
                "provider": "telegram",
                "user_id": "user-1",
                "chat_id": "chat-1",
                "version": "2",
            },
            {
                "type": "message",
                "id": msg_id,
                "role": "user",
                "content": "Remember that Sarah is my sister",
                "created_at": datetime.now(UTC).isoformat(),
                "token_count": 10,
                "username": "david",
                "display_name": "David Cramer",
                "user_id": "user-1",
            },
        ]
        context_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        # Configure mock extractor to return a fact
        mock_extractor.extract_from_conversation.return_value = [
            ExtractedFact(
                content="Sarah is David's sister",
                subjects=["Sarah"],
                shared=False,
                confidence=0.95,
                memory_type=MemoryType.RELATIONSHIP,
                sensitivity=Sensitivity.PUBLIC,
                portable=True,
                speaker="david",
            )
        ]

        server = MockRPCServer()
        register_memory_methods(
            server,  # type: ignore[arg-type]
            memory_manager,
            person_manager=memory_manager,
            memory_extractor=mock_extractor,
            sessions_path=tmp_path / "sessions",
        )

        handler = server.methods["memory.extract"]
        result = await handler(
            {
                "message_id": msg_id,
                "provider": "telegram",
                "user_id": "user-1",
                "chat_id": "chat-1",
                "source_username": "david",
                "source_display_name": "David Cramer",
            }
        )

        assert result["stored"] == 1
        mock_extractor.extract_from_conversation.assert_awaited_once()

        # Verify the memory was stored
        memories = await memory_manager.list_memories(owner_user_id="user-1")
        assert len(memories) >= 1
        assert any("Sarah" in m.content for m in memories)

    async def test_extract_uses_message_author_info(
        self, memory_manager, mock_extractor, tmp_path
    ):
        """Test that extract uses author info from the MessageEntry."""
        import json
        from datetime import UTC, datetime

        from ash.sessions.types import session_key

        key = session_key("telegram", "chat-1", "user-1")
        session_dir = tmp_path / "sessions" / key
        session_dir.mkdir(parents=True)

        msg_id = "test-msg-456"
        context_file = session_dir / "context.jsonl"
        entries = [
            {
                "type": "session",
                "id": "session-1",
                "created_at": datetime.now(UTC).isoformat(),
                "provider": "telegram",
                "user_id": "user-1",
                "chat_id": "chat-1",
                "version": "2",
            },
            {
                "type": "message",
                "id": msg_id,
                "role": "user",
                "content": "I live in Seattle",
                "created_at": datetime.now(UTC).isoformat(),
                "token_count": 10,
                "username": "bob",
                "display_name": "Bob Smith",
                "user_id": "user-1",
            },
        ]
        context_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        mock_extractor.extract_from_conversation.return_value = []

        server = MockRPCServer()
        register_memory_methods(
            server,  # type: ignore[arg-type]
            memory_manager,
            person_manager=memory_manager,
            memory_extractor=mock_extractor,
            sessions_path=tmp_path / "sessions",
        )

        handler = server.methods["memory.extract"]
        await handler(
            {
                "message_id": msg_id,
                "provider": "telegram",
                "user_id": "user-1",
                "chat_id": "chat-1",
            }
        )

        # Verify extract_from_conversation was called with speaker info from message
        call_kwargs = mock_extractor.extract_from_conversation.call_args
        speaker_info = call_kwargs.kwargs.get("speaker_info") or call_kwargs[1].get(
            "speaker_info"
        )
        assert speaker_info is not None
        assert speaker_info.username == "bob"
        assert speaker_info.display_name == "Bob Smith"
