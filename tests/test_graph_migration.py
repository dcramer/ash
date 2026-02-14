"""Tests for graph filesystem and data migration."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ash.graph.migration import migrate_data_to_graph, migrate_filesystem
from ash.graph.types import ChatEntry, UserEntry
from ash.memory.file_store import FileMemoryStore
from ash.memory.jsonl import TypedJSONL
from ash.memory.types import MemoryType
from ash.people.types import AliasEntry, PersonEntry, RelationshipClaim


class TestMigrateFilesystem:
    def test_no_migration_needed(self, ash_home: Path):
        """No old files to migrate -> returns False."""
        assert migrate_filesystem() is False

    def test_migrate_embeddings(self, ash_home: Path):
        """Moves graph/embeddings.jsonl -> index/embeddings.jsonl."""
        graph_dir = ash_home / "graph"
        graph_dir.mkdir()
        old_path = graph_dir / "embeddings.jsonl"
        old_path.write_text('{"memory_id":"m1","embedding":"AAAA"}\n')

        assert migrate_filesystem() is True

        new_path = ash_home / "index" / "embeddings.jsonl"
        assert new_path.exists()
        assert not old_path.exists()
        assert "m1" in new_path.read_text()

    def test_migrate_vector_db(self, ash_home: Path):
        """Moves data/memory.db -> index/vectors.db."""
        data_dir = ash_home / "data"
        data_dir.mkdir()
        old_path = data_dir / "memory.db"
        old_path.write_bytes(b"sqlite3 data")

        assert migrate_filesystem() is True

        new_path = ash_home / "index" / "vectors.db"
        assert new_path.exists()
        assert not old_path.exists()

    def test_migrate_skill_state(self, ash_home: Path):
        """Moves data/skills/ -> skills/state/."""
        old_dir = ash_home / "data" / "skills"
        old_dir.mkdir(parents=True)
        (old_dir / "test_skill.json").write_text(json.dumps({"global": {}}))

        assert migrate_filesystem() is True

        new_dir = ash_home / "skills" / "state"
        assert new_dir.exists()
        assert (new_dir / "test_skill.json").exists()
        assert not old_dir.exists()

    def test_removes_empty_data_dir(self, ash_home: Path):
        """Removes data/ if empty after migrations."""
        data_dir = ash_home / "data"
        data_dir.mkdir()
        old_db = data_dir / "memory.db"
        old_db.write_bytes(b"data")

        migrate_filesystem()

        assert not data_dir.exists()

    def test_keeps_nonempty_data_dir(self, ash_home: Path):
        """Keeps data/ if it still has other files."""
        data_dir = ash_home / "data"
        data_dir.mkdir()
        (data_dir / "other_file.txt").write_text("keep me")

        migrate_filesystem()

        assert data_dir.exists()
        assert (data_dir / "other_file.txt").exists()

    def test_idempotent(self, ash_home: Path):
        """Running twice doesn't error or move files again."""
        graph_dir = ash_home / "graph"
        graph_dir.mkdir()
        old_path = graph_dir / "embeddings.jsonl"
        old_path.write_text('{"memory_id":"m1","embedding":"AAAA"}\n')

        assert migrate_filesystem() is True
        assert migrate_filesystem() is False  # Already migrated

    def test_all_migrations_at_once(self, ash_home: Path):
        """All three migrations happen in one call."""
        # Set up old layout
        graph_dir = ash_home / "graph"
        graph_dir.mkdir()
        (graph_dir / "embeddings.jsonl").write_text('{"memory_id":"m1"}\n')

        data_dir = ash_home / "data"
        data_dir.mkdir()
        (data_dir / "memory.db").write_bytes(b"sqlite3")

        skills_dir = data_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "s1.json").write_text("{}")

        assert migrate_filesystem() is True

        # Verify new layout
        assert (ash_home / "index" / "embeddings.jsonl").exists()
        assert (ash_home / "index" / "vectors.db").exists()
        assert (ash_home / "skills" / "state" / "s1.json").exists()
        assert not data_dir.exists()


class TestMigrateDataToGraph:
    """Tests for data migration: extracting User/Chat nodes from memories."""

    @pytest.fixture
    def graph_dir(self, ash_home: Path) -> Path:
        """Create graph directory."""
        d = ash_home / "graph"
        d.mkdir(exist_ok=True)
        return d

    @pytest.fixture
    def memory_store(self, ash_home: Path) -> FileMemoryStore:
        """Create a FileMemoryStore pointing at temp dir."""
        return FileMemoryStore()

    @pytest.fixture
    def user_jsonl(self, graph_dir: Path) -> TypedJSONL[UserEntry]:
        return TypedJSONL(graph_dir / "users.jsonl", UserEntry)

    @pytest.fixture
    def chat_jsonl(self, graph_dir: Path) -> TypedJSONL[ChatEntry]:
        return TypedJSONL(graph_dir / "chats.jsonl", ChatEntry)

    @pytest.fixture
    def people_jsonl(self, graph_dir: Path) -> TypedJSONL[PersonEntry]:
        return TypedJSONL(graph_dir / "people.jsonl", PersonEntry)

    async def test_no_memories_no_migration(
        self, memory_store, user_jsonl, chat_jsonl, people_jsonl
    ):
        """Empty memory store produces no users or chats."""
        users, chats = await migrate_data_to_graph(
            memory_store, user_jsonl, chat_jsonl, people_jsonl
        )
        assert users == 0
        assert chats == 0

    async def test_extracts_users_from_memories(
        self, memory_store, user_jsonl, chat_jsonl, people_jsonl
    ):
        """Creates UserEntry for each unique owner_user_id."""
        await memory_store.add_memory(
            content="test 1",
            memory_type=MemoryType.KNOWLEDGE,
            owner_user_id="100",
            source_username="alice",
            source_display_name="Alice Smith",
        )
        await memory_store.add_memory(
            content="test 2",
            memory_type=MemoryType.KNOWLEDGE,
            owner_user_id="200",
            source_username="bob",
        )

        users, chats = await migrate_data_to_graph(
            memory_store, user_jsonl, chat_jsonl, people_jsonl
        )

        assert users == 2
        assert chats == 0

        all_users = await user_jsonl.load_all()
        assert len(all_users) == 2
        provider_ids = {u.provider_id for u in all_users}
        assert provider_ids == {"100", "200"}

        alice = next(u for u in all_users if u.provider_id == "100")
        assert alice.username == "alice"
        assert alice.display_name == "Alice Smith"

    async def test_extracts_chats_from_memories(
        self, memory_store, user_jsonl, chat_jsonl, people_jsonl
    ):
        """Creates ChatEntry for each unique chat_id."""
        await memory_store.add_memory(
            content="group fact",
            memory_type=MemoryType.KNOWLEDGE,
            chat_id="chat-100",
        )
        await memory_store.add_memory(
            content="another group fact",
            memory_type=MemoryType.KNOWLEDGE,
            chat_id="chat-200",
        )
        # Duplicate chat_id should not create duplicate
        await memory_store.add_memory(
            content="same chat",
            memory_type=MemoryType.KNOWLEDGE,
            chat_id="chat-100",
        )

        users, chats = await migrate_data_to_graph(
            memory_store, user_jsonl, chat_jsonl, people_jsonl
        )

        assert chats == 2
        all_chats = await chat_jsonl.load_all()
        assert len(all_chats) == 2
        provider_ids = {c.provider_id for c in all_chats}
        assert provider_ids == {"chat-100", "chat-200"}

    async def test_links_user_to_self_person(
        self, memory_store, user_jsonl, chat_jsonl, people_jsonl
    ):
        """Links UserEntry.person_id to self-person via username match."""
        now = datetime.now(UTC)
        # Create a self-person with alias
        person = PersonEntry(
            id="p1",
            name="Alice",
            created_at=now,
            aliases=[AliasEntry(value="alice")],
            relationships=[RelationshipClaim(relationship="self", stated_by="100")],
        )
        await people_jsonl.append(person)

        # Create memory with matching username
        await memory_store.add_memory(
            content="test",
            memory_type=MemoryType.KNOWLEDGE,
            owner_user_id="100",
            source_username="alice",
        )

        users, _ = await migrate_data_to_graph(
            memory_store, user_jsonl, chat_jsonl, people_jsonl
        )

        assert users == 1
        all_users = await user_jsonl.load_all()
        assert all_users[0].person_id == "p1"

    async def test_skips_archived_memories(
        self, memory_store, user_jsonl, chat_jsonl, people_jsonl
    ):
        """Archived memories are not used for user/chat extraction."""
        # Add then archive a memory
        entry = await memory_store.add_memory(
            content="archived",
            memory_type=MemoryType.KNOWLEDGE,
            owner_user_id="100",
        )
        await memory_store.delete_memory(entry.id)

        users, chats = await migrate_data_to_graph(
            memory_store, user_jsonl, chat_jsonl, people_jsonl
        )
        assert users == 0
        assert chats == 0

    async def test_idempotent_skips_if_users_exist(
        self, ash_home, memory_store, user_jsonl, chat_jsonl, people_jsonl
    ):
        """Skips migration if users.jsonl already exists."""
        await memory_store.add_memory(
            content="test",
            memory_type=MemoryType.KNOWLEDGE,
            owner_user_id="100",
        )

        # Create users.jsonl to simulate previous migration
        users_path = ash_home / "graph" / "users.jsonl"
        users_path.write_text("")

        users, chats = await migrate_data_to_graph(
            memory_store, user_jsonl, chat_jsonl, people_jsonl
        )
        assert users == 0
        assert chats == 0

    async def test_merges_username_from_multiple_memories(
        self, memory_store, user_jsonl, chat_jsonl, people_jsonl
    ):
        """Multiple memories for same owner_user_id merge username/display_name."""
        # First memory has username but no display name
        await memory_store.add_memory(
            content="test 1",
            memory_type=MemoryType.KNOWLEDGE,
            owner_user_id="100",
            source_username="alice",
        )
        # Second memory has display name but no username
        await memory_store.add_memory(
            content="test 2",
            memory_type=MemoryType.KNOWLEDGE,
            owner_user_id="100",
            source_display_name="Alice Smith",
        )

        users, _ = await migrate_data_to_graph(
            memory_store, user_jsonl, chat_jsonl, people_jsonl
        )

        assert users == 1
        all_users = await user_jsonl.load_all()
        assert all_users[0].username == "alice"
        assert all_users[0].display_name == "Alice Smith"
