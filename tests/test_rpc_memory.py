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
from ash.store.types import ExtractedFact, MemoryType, Sensitivity, get_assertion


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
    index.search = MagicMock(return_value=[])
    index.add = MagicMock()
    index.remove = MagicMock()
    index.save = AsyncMock()
    index.get_ids = MagicMock(return_value=set())
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

    async def test_add_accepts_structured_assertion(self, rpc_server, memory_manager):
        """memory.add accepts assertion fields and persists assertion metadata."""
        person = await memory_manager.create_person(created_by="user-1", name="Bob")

        handler = rpc_server.methods["memory.add"]
        result = await handler(
            {
                "content": "Bob likes mountain biking",
                "user_id": "user-1",
                "assertion_kind": "person_fact",
                "assertion_subject_ids": [person.id],
                "speaker_person_id": person.id,
                "predicates": [
                    {
                        "name": "describes",
                        "object_type": "text",
                        "value": "Bob likes mountain biking",
                    }
                ],
            }
        )

        memory = await memory_manager.get_memory(result["id"])
        assert memory is not None
        assertion = get_assertion(memory)
        assert assertion is not None
        assert assertion.assertion_kind.value == "person_fact"
        assert assertion.subjects == [person.id]
        assert assertion.speaker_person_id == person.id

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

    async def test_extract_from_messages_requires_provider(self, rpc_server):
        """Explicit-message extraction requires provider for chat provenance."""
        handler = rpc_server.methods["memory.extract_from_messages"]

        with pytest.raises(ValueError, match="provider is required"):
            await handler(
                {
                    "messages": [{"role": "user", "content": "Remember this"}],
                }
            )

    async def test_extract_from_messages_requires_messages(self, rpc_server):
        """Explicit-message extraction requires a non-empty messages list."""
        handler = rpc_server.methods["memory.extract_from_messages"]

        with pytest.raises(ValueError, match="messages must be a non-empty list"):
            await handler({"provider": "telegram", "messages": []})

    async def test_extract_from_messages_runs_pipeline(
        self, rpc_server, memory_manager, mock_extractor
    ):
        """Explicit-message extraction stores facts without session file lookups."""
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

        handler = rpc_server.methods["memory.extract_from_messages"]
        result = await handler(
            {
                "provider": "telegram",
                "user_id": "user-1",
                "chat_id": "chat-1",
                "messages": [
                    {
                        "id": "m-1",
                        "role": "user",
                        "content": "Remember that Sarah is my sister",
                        "user_id": "user-1",
                        "username": "david",
                        "display_name": "David Cramer",
                    }
                ],
            }
        )

        assert result["stored"] == 1
        mock_extractor.extract_from_conversation.assert_awaited_once()

        memories = await memory_manager.list_memories(owner_user_id="user-1")
        assert len(memories) >= 1
        assert any("Sarah" in m.content for m in memories)


class TestRPCDMSourceFiltering:
    """Tests for DM-sourced memory filtering in group chat RPC calls."""

    def _link_to_dm(self, graph, memory_id: str, chat_node_id: str = "dm-chat-1"):
        """Create a LEARNED_IN edge from memory to chat."""
        from ash.graph.edges import create_learned_in_edge

        edge = create_learned_in_edge(memory_id, chat_node_id)
        graph.add_edge(edge)

    async def test_search_filters_dm_sourced_in_group(self, rpc_server, memory_manager):
        """DM-sourced memories should be filtered out when searching in a group chat."""
        from ash.store.types import ChatEntry

        # Create a private chat node
        memory_manager.graph.chats["dm-chat-1"] = ChatEntry(
            id="dm-chat-1",
            provider="telegram",
            provider_id="dm-123",
            chat_type="private",
        )

        # Add a memory and link it to the DM chat via LEARNED_IN edge
        mem = await memory_manager.add_memory(
            content="Secret plan from DM",
            owner_user_id="user-1",
        )
        self._link_to_dm(memory_manager.graph, mem.id)

        # Mock vector search to return this memory
        memory_manager._index.search = MagicMock(return_value=[(mem.id, 0.95)])

        handler = rpc_server.methods["memory.search"]

        # In a group chat, the DM-sourced memory should be filtered out
        results = await handler(
            {
                "query": "secret plan",
                "user_id": "user-1",
                "chat_type": "group",
            }
        )
        assert len(results) == 0

    async def test_search_allows_dm_sourced_in_dm(self, rpc_server, memory_manager):
        """DM-sourced memories should NOT be filtered when searching in a DM."""
        from ash.store.types import ChatEntry

        memory_manager.graph.chats["dm-chat-1"] = ChatEntry(
            id="dm-chat-1",
            provider="telegram",
            provider_id="dm-123",
            chat_type="private",
        )

        mem = await memory_manager.add_memory(
            content="Secret plan from DM",
            owner_user_id="user-1",
        )
        self._link_to_dm(memory_manager.graph, mem.id)

        memory_manager._index.search = MagicMock(return_value=[(mem.id, 0.95)])

        handler = rpc_server.methods["memory.search"]

        # In a private chat, the DM-sourced memory should be visible
        results = await handler(
            {
                "query": "secret plan",
                "user_id": "user-1",
                "chat_type": "private",
                "chat_id": "dm-123",
            }
        )
        assert len(results) == 1
        assert results[0]["content"] == "Secret plan from DM"

    async def test_search_filters_dm_sourced_in_other_dm(
        self, rpc_server, memory_manager
    ):
        """DM-sourced memories should be excluded in a different DM chat."""
        from ash.store.types import ChatEntry

        memory_manager.graph.chats["dm-chat-1"] = ChatEntry(
            id="dm-chat-1",
            provider="telegram",
            provider_id="dm-123",
            chat_type="private",
        )

        mem = await memory_manager.add_memory(
            content="Secret plan from DM",
            owner_user_id="user-1",
        )
        self._link_to_dm(memory_manager.graph, mem.id)

        memory_manager._index.search = MagicMock(return_value=[(mem.id, 0.95)])

        handler = rpc_server.methods["memory.search"]
        results = await handler(
            {
                "query": "secret plan",
                "user_id": "user-1",
                "chat_type": "private",
                "chat_id": "dm-999",
            }
        )
        assert len(results) == 0

    async def test_search_no_chat_type_passes_through(self, rpc_server, memory_manager):
        """Without chat_type, no DM filtering is applied."""
        from ash.store.types import ChatEntry

        memory_manager.graph.chats["dm-chat-1"] = ChatEntry(
            id="dm-chat-1",
            provider="telegram",
            provider_id="dm-123",
            chat_type="private",
        )

        mem = await memory_manager.add_memory(
            content="Secret plan from DM",
            owner_user_id="user-1",
        )
        self._link_to_dm(memory_manager.graph, mem.id)

        memory_manager._index.search = MagicMock(return_value=[(mem.id, 0.95)])

        handler = rpc_server.methods["memory.search"]

        # Without chat_type, memories should pass through
        results = await handler(
            {
                "query": "secret plan",
                "user_id": "user-1",
            }
        )
        assert len(results) == 1

    async def test_search_chat_id_without_chat_type_fails_closed(
        self, rpc_server, memory_manager
    ):
        """Chat-scoped requests without chat_type should fail closed."""
        from ash.store.types import ChatEntry

        memory_manager.graph.chats["dm-chat-1"] = ChatEntry(
            id="dm-chat-1",
            provider="telegram",
            provider_id="dm-123",
            chat_type="private",
        )

        mem = await memory_manager.add_memory(
            content="Secret plan from DM",
            owner_user_id="user-1",
        )
        self._link_to_dm(memory_manager.graph, mem.id)

        memory_manager._index.search = MagicMock(return_value=[(mem.id, 0.95)])

        handler = rpc_server.methods["memory.search"]
        results = await handler(
            {
                "query": "secret plan",
                "user_id": "user-1",
                "chat_id": "dm-123",
            }
        )
        assert len(results) == 0

    async def test_list_filters_dm_sourced_in_group(self, rpc_server, memory_manager):
        """DM-sourced memories should be filtered from list in group chats."""
        from ash.graph.edges import create_learned_in_edge
        from ash.store.types import ChatEntry

        memory_manager.graph.chats["dm-chat-1"] = ChatEntry(
            id="dm-chat-1",
            provider="telegram",
            provider_id="dm-123",
            chat_type="private",
        )

        mem = await memory_manager.add_memory(
            content="Secret from DM",
            owner_user_id="user-1",
        )
        self._link_to_dm(memory_manager.graph, mem.id)

        # Add a memory with group chat provenance
        group_chat = ChatEntry(
            id="group-chat-1",
            provider="telegram",
            provider_id="group-456",
            chat_type="group",
        )
        memory_manager.graph.add_chat(group_chat)
        mem2 = await memory_manager.add_memory(
            content="Public knowledge",
            owner_user_id="user-1",
        )
        memory_manager.graph.add_edge(create_learned_in_edge(mem2.id, group_chat.id))

        handler = rpc_server.methods["memory.list"]

        results = await handler(
            {
                "user_id": "user-1",
                "chat_type": "group",
            }
        )
        assert len(results) == 1
        assert results[0]["content"] == "Public knowledge"

    async def test_list_filters_dm_sourced_in_other_dm(
        self, rpc_server, memory_manager
    ):
        """DM-sourced memories should be filtered from list in a different DM chat."""
        from ash.store.types import ChatEntry

        memory_manager.graph.chats["dm-chat-1"] = ChatEntry(
            id="dm-chat-1",
            provider="telegram",
            provider_id="dm-123",
            chat_type="private",
        )

        mem = await memory_manager.add_memory(
            content="Secret from DM",
            owner_user_id="user-1",
        )
        self._link_to_dm(memory_manager.graph, mem.id)

        handler = rpc_server.methods["memory.list"]
        results = await handler(
            {
                "user_id": "user-1",
                "chat_type": "private",
                "chat_id": "dm-999",
            }
        )
        assert len(results) == 0

    async def test_list_resolves_chat_type_from_provider_context(
        self, rpc_server, memory_manager
    ):
        """List should resolve chat_type from provider+chat_id like search does."""
        from ash.graph.edges import create_learned_in_edge
        from ash.store.types import ChatEntry

        group_chat = ChatEntry(
            id="group-chat-1",
            provider="telegram",
            provider_id="group-456",
            chat_type="group",
        )
        memory_manager.graph.add_chat(group_chat)

        mem = await memory_manager.add_memory(
            content="Group-sourced fact",
            owner_user_id="user-1",
        )
        memory_manager.graph.add_edge(create_learned_in_edge(mem.id, group_chat.id))

        handler = rpc_server.methods["memory.list"]
        results = await handler(
            {
                "user_id": "user-1",
                "provider": "telegram",
                "chat_id": "group-456",
            }
        )

        assert len(results) == 1
        assert results[0]["content"] == "Group-sourced fact"

    async def test_search_filters_missing_provenance_memories_in_group(
        self, rpc_server, memory_manager
    ):
        """Memories missing LEARNED_IN provenance should be excluded in group chats."""
        # Add a memory with no LEARNED_IN edge (missing provenance)
        mem = await memory_manager.add_memory(
            content="Fact missing provenance",
            owner_user_id="user-1",
        )

        memory_manager._index.search = MagicMock(return_value=[(mem.id, 0.90)])

        handler = rpc_server.methods["memory.search"]
        results = await handler(
            {
                "query": "missing provenance",
                "user_id": "user-1",
                "chat_type": "group",
            }
        )
        assert len(results) == 0

    async def test_list_filters_missing_provenance_memories_in_group(
        self, rpc_server, memory_manager
    ):
        """Memories missing LEARNED_IN provenance should be excluded from list in group chats."""
        from ash.graph.edges import create_learned_in_edge
        from ash.store.types import ChatEntry

        # Create a group chat node
        group_chat = ChatEntry(
            id="group-chat-1",
            provider="telegram",
            provider_id="group-456",
            chat_type="group",
        )
        memory_manager.graph.add_chat(group_chat)

        # Memory missing LEARNED_IN provenance
        await memory_manager.add_memory(
            content="Fact missing provenance",
            owner_user_id="user-1",
        )

        # Memory with LEARNED_IN edge to group chat
        mem2 = await memory_manager.add_memory(
            content="Group-sourced fact",
            owner_user_id="user-1",
        )
        memory_manager.graph.add_edge(create_learned_in_edge(mem2.id, group_chat.id))

        handler = rpc_server.methods["memory.list"]
        results = await handler(
            {
                "user_id": "user-1",
                "chat_type": "group",
            }
        )
        assert len(results) == 1
        assert results[0]["content"] == "Group-sourced fact"


class TestRPCMemoryListTrust:
    """Tests for trust field in memory.list response."""

    async def test_list_includes_trust_field(self, rpc_server, memory_manager):
        """memory.list response should include trust classification for each memory."""
        await memory_manager.add_memory(
            content="A simple fact",
            owner_user_id="user-1",
        )

        handler = rpc_server.methods["memory.list"]
        results = await handler({"user_id": "user-1"})

        assert len(results) == 1
        assert "trust" in results[0]
        # No STATED_BY edge → unknown trust
        assert results[0]["trust"] == "unknown"

    async def test_list_trust_reflects_graph_edges(self, rpc_server, memory_manager):
        """Trust should reflect STATED_BY/ABOUT graph edges."""
        from ash.graph.edges import create_about_edge, create_stated_by_edge

        # Create a person
        person = await memory_manager.create_person(
            created_by="alice", name="Alice", aliases=["alice"]
        )

        # Memory stated by Alice about Alice (fact)
        mem_fact = await memory_manager.add_memory(
            content="Alice likes hiking",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        memory_manager.graph.add_edge(create_stated_by_edge(mem_fact.id, person.id))
        memory_manager.graph.add_edge(create_about_edge(mem_fact.id, person.id))

        # Create another person for hearsay
        other = await memory_manager.create_person(
            created_by="alice", name="Bob", aliases=["bob"]
        )

        # Memory stated by Bob about Alice (hearsay)
        mem_hearsay = await memory_manager.add_memory(
            content="Alice likes swimming",
            owner_user_id="user-1",
            subject_person_ids=[person.id],
        )
        memory_manager.graph.add_edge(create_stated_by_edge(mem_hearsay.id, other.id))
        memory_manager.graph.add_edge(create_about_edge(mem_hearsay.id, person.id))

        handler = rpc_server.methods["memory.list"]
        results = await handler({"user_id": "user-1"})

        trust_by_content = {r["content"]: r["trust"] for r in results}
        assert trust_by_content["Alice likes hiking"] == "fact"
        assert trust_by_content["Alice likes swimming"] == "hearsay"


class TestRPCThisChatFiltering:
    """Tests for --this-chat filtering in search and list."""

    async def test_search_this_chat_returns_only_chat_memories(
        self, rpc_server, memory_manager
    ):
        """With this_chat=True, only memories learned in the current chat are returned."""
        from ash.graph.edges import create_learned_in_edge
        from ash.store.types import ChatEntry

        # Register the chat in graph
        chat = ChatEntry(
            id="graph-chat-1",
            provider="telegram",
            provider_id="chat-100",
            chat_type="private",
        )
        memory_manager.graph.add_chat(chat)

        # Memory learned in this chat
        mem1 = await memory_manager.add_memory(
            content="Learned here",
            owner_user_id="user-1",
        )
        memory_manager.graph.add_edge(create_learned_in_edge(mem1.id, chat.id))

        # Memory from another chat
        mem2 = await memory_manager.add_memory(
            content="Learned elsewhere",
            owner_user_id="user-1",
        )

        memory_manager._index.search = MagicMock(
            return_value=[(mem1.id, 0.90), (mem2.id, 0.85)]
        )

        handler = rpc_server.methods["memory.search"]
        results = await handler(
            {
                "query": "learned",
                "user_id": "user-1",
                "provider": "telegram",
                "chat_id": "chat-100",
                "this_chat": True,
            }
        )
        assert len(results) == 1
        assert results[0]["content"] == "Learned here"

    async def test_search_without_this_chat_returns_all(
        self, rpc_server, memory_manager
    ):
        """Without this_chat, private chat visibility still filters unknown provenance."""
        from ash.graph.edges import create_learned_in_edge
        from ash.store.types import ChatEntry

        chat = ChatEntry(
            id="graph-chat-2",
            provider="telegram",
            provider_id="chat-200",
            chat_type="private",
        )
        memory_manager.graph.add_chat(chat)

        mem1 = await memory_manager.add_memory(
            content="From this chat",
            owner_user_id="user-1",
        )
        memory_manager.graph.add_edge(create_learned_in_edge(mem1.id, chat.id))

        mem2 = await memory_manager.add_memory(
            content="From other chat",
            owner_user_id="user-1",
        )

        memory_manager._index.search = MagicMock(
            return_value=[(mem1.id, 0.90), (mem2.id, 0.85)]
        )

        handler = rpc_server.methods["memory.search"]
        results = await handler(
            {
                "query": "chat",
                "user_id": "user-1",
                "provider": "telegram",
                "chat_id": "chat-200",
            }
        )
        assert len(results) == 1
        assert results[0]["content"] == "From this chat"

    async def test_list_this_chat_filters(self, rpc_server, memory_manager):
        """memory.list with this_chat=True only returns memories learned in current chat."""
        from ash.graph.edges import create_learned_in_edge
        from ash.store.types import ChatEntry

        chat = ChatEntry(
            id="graph-chat-3",
            provider="telegram",
            provider_id="chat-300",
            chat_type="private",
        )
        memory_manager.graph.add_chat(chat)

        mem1 = await memory_manager.add_memory(
            content="This chat memory",
            owner_user_id="user-1",
        )
        memory_manager.graph.add_edge(create_learned_in_edge(mem1.id, chat.id))

        await memory_manager.add_memory(
            content="Other chat memory",
            owner_user_id="user-1",
        )

        handler = rpc_server.methods["memory.list"]
        results = await handler(
            {
                "user_id": "user-1",
                "provider": "telegram",
                "chat_id": "chat-300",
                "this_chat": True,
            }
        )
        assert len(results) == 1
        assert results[0]["content"] == "This chat memory"

    async def test_this_chat_excludes_missing_provenance_memories(
        self, rpc_server, memory_manager
    ):
        """Memories without LEARNED_IN edges are excluded when this_chat is active."""
        from ash.graph.edges import create_learned_in_edge
        from ash.store.types import ChatEntry

        chat = ChatEntry(
            id="graph-chat-4",
            provider="telegram",
            provider_id="chat-400",
            chat_type="private",
        )
        memory_manager.graph.add_chat(chat)

        # Memory missing LEARNED_IN provenance
        mem_missing_provenance = await memory_manager.add_memory(
            content="Memory missing provenance",
            owner_user_id="user-1",
        )

        # Memory learned in this chat
        mem_here = await memory_manager.add_memory(
            content="Learned in this chat",
            owner_user_id="user-1",
        )
        memory_manager.graph.add_edge(create_learned_in_edge(mem_here.id, chat.id))

        memory_manager._index.search = MagicMock(
            return_value=[(mem_missing_provenance.id, 0.90), (mem_here.id, 0.85)]
        )

        handler = rpc_server.methods["memory.search"]
        results = await handler(
            {
                "query": "memory",
                "user_id": "user-1",
                "provider": "telegram",
                "chat_id": "chat-400",
                "this_chat": True,
            }
        )
        assert len(results) == 1
        assert results[0]["content"] == "Learned in this chat"

    async def test_search_this_chat_fails_closed_without_resolved_chat(
        self, rpc_server, memory_manager
    ):
        """this_chat search returns no results when chat provenance can't be resolved."""
        mem = await memory_manager.add_memory(
            content="Unscoped candidate",
            owner_user_id="user-1",
        )
        memory_manager._index.search = MagicMock(return_value=[(mem.id, 0.90)])

        handler = rpc_server.methods["memory.search"]
        results = await handler(
            {
                "query": "candidate",
                "user_id": "user-1",
                "provider": "telegram",
                "chat_id": "unknown-chat",
                "this_chat": True,
            }
        )
        assert results == []

    async def test_list_this_chat_fails_closed_without_resolved_chat(
        self, rpc_server, memory_manager
    ):
        """this_chat list returns no results when chat provenance can't be resolved."""
        await memory_manager.add_memory(
            content="Unscoped candidate",
            owner_user_id="user-1",
        )

        handler = rpc_server.methods["memory.list"]
        results = await handler(
            {
                "user_id": "user-1",
                "provider": "telegram",
                "chat_id": "unknown-chat",
                "this_chat": True,
            }
        )
        assert results == []
