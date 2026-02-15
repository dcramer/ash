"""Tests for self-fact subject attribution.

Tests:
- _ensure_self_person returning person_id (new, existing, dedup merge, no store)
- Doctor command backfilling orphaned self-facts
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.config import AshConfig
from ash.config.models import ModelConfig
from ash.config.workspace import Workspace
from ash.core.agent import Agent
from ash.core.prompt import SystemPromptBuilder
from ash.db.engine import Database
from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.index import VectorIndex
from ash.skills.registry import SkillRegistry
from ash.store.store import Store
from ash.store.types import MemoryType
from ash.tools.executor import ToolExecutor
from ash.tools.registry import ToolRegistry
from tests.conftest import MockLLMProvider

DEFAULT_MODEL_CONFIG = {
    "default": ModelConfig(provider="anthropic", model="claude-test")
}


@pytest.fixture
def mock_index():
    index = MagicMock(spec=VectorIndex)
    index.search = AsyncMock(return_value=[])
    index.add_embedding = AsyncMock()
    index.delete_embedding = AsyncMock()
    index.delete_embeddings = AsyncMock()
    return index


@pytest.fixture
def mock_embedding_generator():
    generator = MagicMock(spec=EmbeddingGenerator)
    generator.embed = AsyncMock(return_value=[0.1] * 1536)
    return generator


@pytest.fixture
async def store(database: Database, mock_index, mock_embedding_generator) -> Store:
    return Store(
        db=database,
        vector_index=mock_index,
        embedding_generator=mock_embedding_generator,
    )


def _make_agent(store: Store, workspace: Workspace) -> Agent:
    registry = ToolRegistry()
    return Agent(
        llm=MockLLMProvider(),
        tool_executor=ToolExecutor(registry),
        prompt_builder=SystemPromptBuilder(
            workspace=workspace,
            tool_registry=registry,
            skill_registry=SkillRegistry(),
            config=AshConfig(workspace=workspace.path, models=DEFAULT_MODEL_CONFIG),
        ),
        graph_store=store,
    )


class TestEnsureSelfPersonReturnsId:
    """Test _ensure_self_person returns person_id."""

    async def test_returns_id_for_new_person(self, store: Store, tmp_path: Path):
        workspace = Workspace(path=tmp_path, soul="test")
        agent = _make_agent(store, workspace)

        person_id = await agent._ensure_self_person(
            user_id="user-1",
            username="notzeeg",
            display_name="David Cramer",
        )

        assert person_id is not None
        person = await store.get_person(person_id)
        assert person is not None
        assert person.name == "David Cramer"

    async def test_returns_id_for_existing_person(self, store: Store, tmp_path: Path):
        workspace = Workspace(path=tmp_path, soul="test")
        agent = _make_agent(store, workspace)

        # Create person first
        created = await store.create_person(
            created_by="user-1",
            name="David Cramer",
            relationship="self",
            aliases=["notzeeg"],
        )

        person_id = await agent._ensure_self_person(
            user_id="user-1",
            username="notzeeg",
            display_name="David Cramer",
        )

        assert person_id == created.id

    async def test_returns_primary_id_after_dedup_merge(
        self, store: Store, tmp_path: Path
    ):
        workspace = Workspace(path=tmp_path, soul="test")
        # Need an LLM for dedup verification
        mock_llm = _make_dedup_llm()
        store.set_llm(mock_llm, "mock-model")
        agent = _make_agent(store, workspace)

        # Create an existing person with same name but no self relationship
        await store.create_person(
            created_by="user-1",
            name="David Cramer",
            relationship="friend",
            aliases=["david"],
        )

        # _ensure_self_person creates a new person then dedup should merge
        person_id = await agent._ensure_self_person(
            user_id="user-1",
            username="notzeeg",
            display_name="David Cramer",
        )

        assert person_id is not None
        # After merge, the returned id should resolve to a non-merged person
        person = await store.get_person(person_id)
        assert person is not None
        assert person.merged_into is None

    async def test_returns_none_without_people_store(self, tmp_path: Path):
        workspace = Workspace(path=tmp_path, soul="test")
        registry = ToolRegistry()
        agent = Agent(
            llm=MockLLMProvider(),
            tool_executor=ToolExecutor(registry),
            prompt_builder=SystemPromptBuilder(
                workspace=workspace,
                tool_registry=registry,
                skill_registry=SkillRegistry(),
                config=AshConfig(workspace=workspace.path, models=DEFAULT_MODEL_CONFIG),
            ),
            graph_store=None,
        )

        person_id = await agent._ensure_self_person(
            user_id="user-1",
            username="notzeeg",
            display_name="David Cramer",
        )

        assert person_id is None


class TestDoctorSelfFacts:
    """Test memory doctor self_facts command."""

    async def test_fixes_orphaned_self_facts(self, store: Store):
        # Create a self-person
        person = await store.create_person(
            created_by="user-1",
            name="David Cramer",
            relationship="self",
            aliases=["notzeeg"],
        )

        # Create a memory with source_username but no subject_person_ids
        await store.add_memory(
            content="I love Python",
            source="background_extraction",
            memory_type=MemoryType.IDENTITY,
            owner_user_id="user-1",
            source_username="notzeeg",
        )

        from ash.cli.commands.memory.doctor.self_facts import (
            memory_doctor_self_facts,
        )

        await memory_doctor_self_facts(store, force=True)

        memories = await store.list_memories(limit=None)
        assert len(memories) == 1
        assert memories[0].subject_person_ids == [person.id]

    async def test_skips_relationship_type(self, store: Store):
        # Create a self-person
        await store.create_person(
            created_by="user-1",
            name="David Cramer",
            relationship="self",
            aliases=["notzeeg"],
        )

        # Create a RELATIONSHIP memory — should NOT be fixed
        await store.add_memory(
            content="Sarah is David's wife",
            source="background_extraction",
            memory_type=MemoryType.RELATIONSHIP,
            owner_user_id="user-1",
            source_username="notzeeg",
        )

        from ash.cli.commands.memory.doctor.self_facts import (
            memory_doctor_self_facts,
        )

        await memory_doctor_self_facts(store, force=True)

        memories = await store.list_memories(limit=None)
        assert len(memories) == 1
        assert memories[0].subject_person_ids == []

    async def test_skips_memories_with_existing_subjects(self, store: Store):
        # Create a self-person
        await store.create_person(
            created_by="user-1",
            name="David Cramer",
            relationship="self",
            aliases=["notzeeg"],
        )

        # Create a person to use as subject
        other = await store.create_person(
            created_by="user-1",
            name="Sarah",
            relationship="wife",
        )

        # Memory already has subject_person_ids — should not be touched
        await store.add_memory(
            content="Sarah likes hiking",
            source="background_extraction",
            memory_type=MemoryType.KNOWLEDGE,
            owner_user_id="user-1",
            source_username="notzeeg",
            subject_person_ids=[other.id],
        )

        from ash.cli.commands.memory.doctor.self_facts import (
            memory_doctor_self_facts,
        )

        await memory_doctor_self_facts(store, force=True)

        memories = await store.list_memories(limit=None)
        assert len(memories) == 1
        assert memories[0].subject_person_ids == [other.id]

    async def test_matches_by_display_name(self, store: Store):
        # Create a self-person
        person = await store.create_person(
            created_by="user-1",
            name="David Cramer",
            relationship="self",
            aliases=["notzeeg"],
        )

        # Memory with display name as source_username (matches name)
        await store.add_memory(
            content="I work at Sentry",
            source="background_extraction",
            memory_type=MemoryType.IDENTITY,
            owner_user_id="user-1",
            source_username="David Cramer",
        )

        from ash.cli.commands.memory.doctor.self_facts import (
            memory_doctor_self_facts,
        )

        await memory_doctor_self_facts(store, force=True)

        memories = await store.list_memories(limit=None)
        assert len(memories) == 1
        assert memories[0].subject_person_ids == [person.id]


def _make_dedup_llm() -> AsyncMock:
    """Create a mock LLM that approves dedup matches."""
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.message.get_text.return_value = "YES"
    mock_llm.complete.return_value = mock_response
    return mock_llm
