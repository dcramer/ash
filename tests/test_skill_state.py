"""Tests for file-based skill state storage."""

import json
from pathlib import Path

import pytest

from ash.skills.state import SkillStateStore


class TestSkillStateStore:
    """Tests for SkillStateStore file-based storage."""

    @pytest.fixture
    def state_dir(self, tmp_path: Path) -> Path:
        """Create a temporary directory for state files."""
        state_path = tmp_path / "skills"
        state_path.mkdir(parents=True)
        return state_path

    @pytest.fixture
    def store(self, state_dir: Path) -> SkillStateStore:
        """Create a SkillStateStore with temporary directory."""
        return SkillStateStore(base_path=state_dir)

    # Global state tests

    def test_get_nonexistent_returns_none(self, store: SkillStateStore) -> None:
        """Test that getting a nonexistent key returns None."""
        result = store.get("test-skill", "missing-key")
        assert result is None

    def test_set_and_get_global_state(self, store: SkillStateStore) -> None:
        """Test setting and getting global state."""
        store.set("test-skill", "key1", "value1")
        result = store.get("test-skill", "key1")
        assert result == "value1"

    def test_set_overwrites_existing(self, store: SkillStateStore) -> None:
        """Test that setting a key overwrites existing value."""
        store.set("test-skill", "key1", "original")
        store.set("test-skill", "key1", "updated")
        result = store.get("test-skill", "key1")
        assert result == "updated"

    def test_set_complex_values(self, store: SkillStateStore) -> None:
        """Test storing complex JSON-serializable values."""
        complex_value = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "boolean": True,
            "number": 42,
        }
        store.set("test-skill", "complex", complex_value)
        result = store.get("test-skill", "complex")
        assert result == complex_value

    def test_delete_existing_key(self, store: SkillStateStore) -> None:
        """Test deleting an existing key."""
        store.set("test-skill", "key1", "value1")
        deleted = store.delete("test-skill", "key1")
        assert deleted is True
        assert store.get("test-skill", "key1") is None

    def test_delete_nonexistent_returns_false(self, store: SkillStateStore) -> None:
        """Test that deleting nonexistent key returns False."""
        deleted = store.delete("test-skill", "missing-key")
        assert deleted is False

    def test_get_all_global_state(self, store: SkillStateStore) -> None:
        """Test getting all global state for a skill."""
        store.set("test-skill", "key1", "value1")
        store.set("test-skill", "key2", "value2")
        result = store.get_all("test-skill")
        assert result == {"key1": "value1", "key2": "value2"}

    def test_get_all_empty_returns_empty_dict(self, store: SkillStateStore) -> None:
        """Test that get_all returns empty dict for new skill."""
        result = store.get_all("nonexistent-skill")
        assert result == {}

    # User-scoped state tests

    def test_set_and_get_user_state(self, store: SkillStateStore) -> None:
        """Test setting and getting user-scoped state."""
        store.set("test-skill", "key1", "user-value", user_id="user-123")
        result = store.get("test-skill", "key1", user_id="user-123")
        assert result == "user-value"

    def test_user_state_isolated_from_global(self, store: SkillStateStore) -> None:
        """Test that user state is separate from global state."""
        store.set("test-skill", "key1", "global-value")
        store.set("test-skill", "key1", "user-value", user_id="user-123")

        # Global and user state should be independent
        assert store.get("test-skill", "key1") == "global-value"
        assert store.get("test-skill", "key1", user_id="user-123") == "user-value"

    def test_user_state_isolated_between_users(self, store: SkillStateStore) -> None:
        """Test that different users have separate state."""
        store.set("test-skill", "key1", "user1-value", user_id="user-1")
        store.set("test-skill", "key1", "user2-value", user_id="user-2")

        assert store.get("test-skill", "key1", user_id="user-1") == "user1-value"
        assert store.get("test-skill", "key1", user_id="user-2") == "user2-value"

    def test_delete_user_state(self, store: SkillStateStore) -> None:
        """Test deleting user-scoped state."""
        store.set("test-skill", "key1", "value", user_id="user-123")
        deleted = store.delete("test-skill", "key1", user_id="user-123")
        assert deleted is True
        assert store.get("test-skill", "key1", user_id="user-123") is None

    def test_delete_user_state_leaves_global(self, store: SkillStateStore) -> None:
        """Test that deleting user state doesn't affect global."""
        store.set("test-skill", "key1", "global-value")
        store.set("test-skill", "key1", "user-value", user_id="user-123")

        store.delete("test-skill", "key1", user_id="user-123")

        assert store.get("test-skill", "key1") == "global-value"
        assert store.get("test-skill", "key1", user_id="user-123") is None

    def test_get_all_user_state(self, store: SkillStateStore) -> None:
        """Test getting all user-scoped state."""
        store.set("test-skill", "key1", "value1", user_id="user-123")
        store.set("test-skill", "key2", "value2", user_id="user-123")
        store.set("test-skill", "global-key", "global-value")

        result = store.get_all("test-skill", user_id="user-123")
        assert result == {"key1": "value1", "key2": "value2"}

    # Skill isolation tests

    def test_skills_have_separate_state(self, store: SkillStateStore) -> None:
        """Test that different skills have separate state."""
        store.set("skill-a", "key1", "value-a")
        store.set("skill-b", "key1", "value-b")

        assert store.get("skill-a", "key1") == "value-a"
        assert store.get("skill-b", "key1") == "value-b"

    # Clear tests

    def test_clear_removes_all_state(
        self, store: SkillStateStore, state_dir: Path
    ) -> None:
        """Test that clear removes the skill's state file."""
        store.set("test-skill", "key1", "value1")
        store.set("test-skill", "key2", "value2", user_id="user-123")

        store.clear("test-skill")

        assert store.get("test-skill", "key1") is None
        assert store.get("test-skill", "key2", user_id="user-123") is None
        assert not (state_dir / "test-skill.json").exists()

    def test_clear_nonexistent_is_noop(self, store: SkillStateStore) -> None:
        """Test that clearing nonexistent skill doesn't error."""
        store.clear("nonexistent-skill")  # Should not raise

    # File format tests

    def test_state_file_is_valid_json(
        self, store: SkillStateStore, state_dir: Path
    ) -> None:
        """Test that state is stored as valid JSON."""
        store.set("test-skill", "key1", "value1")
        store.set("test-skill", "key2", "value2", user_id="user-123")

        state_file = state_dir / "test-skill.json"
        assert state_file.exists()

        with state_file.open() as f:
            data = json.load(f)

        assert data == {
            "global": {"key1": "value1"},
            "users": {"user-123": {"key2": "value2"}},
        }

    def test_skill_name_sanitization(
        self, store: SkillStateStore, state_dir: Path
    ) -> None:
        """Test that skill names with slashes are sanitized."""
        store.set("namespace/skill", "key1", "value1")

        # Should use underscore instead of slash
        state_file = state_dir / "namespace_skill.json"
        assert state_file.exists()

        # Should still be accessible
        assert store.get("namespace/skill", "key1") == "value1"

    # Error handling tests

    def test_handles_corrupted_json(
        self, store: SkillStateStore, state_dir: Path
    ) -> None:
        """Test that corrupted JSON files are handled gracefully."""
        state_file = state_dir / "test-skill.json"
        state_file.write_text("invalid json {{{")

        # Should return None and not crash
        result = store.get("test-skill", "key1")
        assert result is None

        # Should be able to write new state
        store.set("test-skill", "key1", "value1")
        assert store.get("test-skill", "key1") == "value1"

    def test_handles_missing_structure(
        self, store: SkillStateStore, state_dir: Path
    ) -> None:
        """Test that malformed JSON structure is handled."""
        state_file = state_dir / "test-skill.json"
        state_file.write_text('{"unexpected": "structure"}')

        # Should return None for missing keys
        result = store.get("test-skill", "key1")
        assert result is None

    # Persistence tests

    def test_state_persists_across_instances(self, state_dir: Path) -> None:
        """Test that state persists when creating new store instance."""
        store1 = SkillStateStore(base_path=state_dir)
        store1.set("test-skill", "key1", "value1")

        store2 = SkillStateStore(base_path=state_dir)
        result = store2.get("test-skill", "key1")
        assert result == "value1"
