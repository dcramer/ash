"""Tests for graph filesystem and data migration."""

import json
from pathlib import Path

from ash.store.migration import migrate_filesystem


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
