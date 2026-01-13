"""File-based skill state storage.

Provides persistent key-value storage for skills. Each skill gets its own
JSON file at ~/.ash/data/skills/<skill-name>.json.

Storage format:
{
    "global": {
        "key1": "value1"
    },
    "users": {
        "user-123": {
            "key1": "user-specific-value"
        }
    }
}
"""

import json
import logging
from pathlib import Path
from typing import Any

from ash.config.paths import get_skill_state_path

logger = logging.getLogger(__name__)


class SkillStateStore:
    """File-based state storage for skills.

    Each skill gets a separate JSON file. State can be global or per-user.
    Uses atomic writes (write to temp file, then rename) for safety.
    """

    def __init__(self, base_path: Path | None = None):
        """Initialize skill state store.

        Args:
            base_path: Base directory for state files. Defaults to ~/.ash/data/skills/
        """
        self._base_path = base_path or get_skill_state_path()

    def _get_state_file(self, skill_name: str) -> Path:
        """Get the path to a skill's state file."""
        # Sanitize skill name for filesystem
        safe_name = skill_name.replace("/", "_").replace("\\", "_")
        return self._base_path / f"{safe_name}.json"

    def _load_state(self, skill_name: str) -> dict[str, Any]:
        """Load state from file, returning empty structure if not found."""
        state_file = self._get_state_file(skill_name)
        if not state_file.exists():
            return {"global": {}, "users": {}}

        try:
            with state_file.open("r") as f:
                data = json.load(f)
                # Ensure structure is valid
                if "global" not in data:
                    data["global"] = {}
                if "users" not in data:
                    data["users"] = {}
                return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Failed to load skill state, starting fresh",
                extra={"skill_name": skill_name, "error": str(e)},
            )
            return {"global": {}, "users": {}}

    def _save_state(self, skill_name: str, state: dict[str, Any]) -> None:
        """Save state to file atomically."""
        state_file = self._get_state_file(skill_name)
        state_file.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first, then rename (atomic on POSIX)
        temp_file = state_file.with_suffix(".json.tmp")
        try:
            with temp_file.open("w") as f:
                json.dump(state, f, indent=2, default=str)
            temp_file.replace(state_file)
        except OSError:
            # Clean up temp file on failure
            if temp_file.exists():
                temp_file.unlink()
            raise

    def get(
        self,
        skill_name: str,
        key: str,
        user_id: str | None = None,
    ) -> Any | None:
        """Get a skill state value.

        Args:
            skill_name: Name of the skill.
            key: State key.
            user_id: User ID for user-scoped state (None for global).

        Returns:
            State value or None if not found.
        """
        state = self._load_state(skill_name)

        if user_id:
            user_state = state.get("users", {}).get(user_id, {})
            return user_state.get(key)
        else:
            return state.get("global", {}).get(key)

    def set(
        self,
        skill_name: str,
        key: str,
        value: Any,
        user_id: str | None = None,
    ) -> None:
        """Set a skill state value.

        Args:
            skill_name: Name of the skill.
            key: State key.
            value: State value (must be JSON-serializable).
            user_id: User ID for user-scoped state (None for global).
        """
        state = self._load_state(skill_name)

        if user_id:
            if user_id not in state["users"]:
                state["users"][user_id] = {}
            state["users"][user_id][key] = value
        else:
            state["global"][key] = value

        self._save_state(skill_name, state)

    def delete(
        self,
        skill_name: str,
        key: str,
        user_id: str | None = None,
    ) -> bool:
        """Delete a skill state value.

        Args:
            skill_name: Name of the skill.
            key: State key.
            user_id: User ID for user-scoped state (None for global).

        Returns:
            True if deleted, False if key was not found.
        """
        state = self._load_state(skill_name)

        if user_id:
            user_state = state.get("users", {}).get(user_id, {})
            if key not in user_state:
                return False
            del state["users"][user_id][key]
            # Clean up empty user dict
            if not state["users"][user_id]:
                del state["users"][user_id]
        else:
            if key not in state.get("global", {}):
                return False
            del state["global"][key]

        self._save_state(skill_name, state)
        return True

    def get_all(
        self,
        skill_name: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Get all state values for a skill.

        Args:
            skill_name: Name of the skill.
            user_id: User ID for user-scoped state (None for global).

        Returns:
            Dict mapping keys to values.
        """
        state = self._load_state(skill_name)

        if user_id:
            return dict(state.get("users", {}).get(user_id, {}))
        else:
            return dict(state.get("global", {}))

    def clear(self, skill_name: str) -> None:
        """Clear all state for a skill.

        Args:
            skill_name: Name of the skill.
        """
        state_file = self._get_state_file(skill_name)
        if state_file.exists():
            state_file.unlink()
