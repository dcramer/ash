"""File-based skill state storage."""

import json
import logging
from pathlib import Path
from typing import Any

from ash.config.paths import get_skill_state_path

logger = logging.getLogger(__name__)


class SkillStateStore:
    """File-based state storage for skills with global and per-user scopes."""

    def __init__(self, base_path: Path | None = None):
        self._base_path = base_path or get_skill_state_path()

    def _get_state_file(self, skill_name: str) -> Path:
        safe_name = skill_name.replace("/", "_").replace("\\", "_")
        return self._base_path / f"{safe_name}.json"

    def _load_state(self, skill_name: str) -> dict[str, Any]:
        state_file = self._get_state_file(skill_name)
        if not state_file.exists():
            return {"global": {}, "users": {}}

        try:
            with state_file.open("r") as f:
                data = json.load(f)
                data.setdefault("global", {})
                data.setdefault("users", {})
                return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "skill_state_load_failed",
                extra={"skill.name": skill_name, "error.message": str(e)},
            )
            return {"global": {}, "users": {}}

    def _save_state(self, skill_name: str, state: dict[str, Any]) -> None:
        state_file = self._get_state_file(skill_name)
        state_file.parent.mkdir(parents=True, exist_ok=True)

        temp_file = state_file.with_suffix(".json.tmp")
        try:
            with temp_file.open("w") as f:
                json.dump(state, f, indent=2, default=str)
            temp_file.replace(state_file)
        except OSError:
            if temp_file.exists():
                temp_file.unlink()
            raise

    def _get_scope(self, state: dict[str, Any], user_id: str | None) -> dict[str, Any]:
        if user_id:
            return state["users"].setdefault(user_id, {})
        return state["global"]

    def get(self, skill_name: str, key: str, user_id: str | None = None) -> Any | None:
        state = self._load_state(skill_name)
        scope = state["users"].get(user_id, {}) if user_id else state["global"]
        return scope.get(key)

    def set(
        self, skill_name: str, key: str, value: Any, user_id: str | None = None
    ) -> None:
        state = self._load_state(skill_name)
        self._get_scope(state, user_id)[key] = value
        self._save_state(skill_name, state)

    def delete(self, skill_name: str, key: str, user_id: str | None = None) -> bool:
        state = self._load_state(skill_name)
        scope = state["users"].get(user_id, {}) if user_id else state["global"]

        if key not in scope:
            return False

        del scope[key]
        if user_id and not scope:
            del state["users"][user_id]

        self._save_state(skill_name, state)
        return True

    def get_all(self, skill_name: str, user_id: str | None = None) -> dict[str, Any]:
        state = self._load_state(skill_name)
        scope = state["users"].get(user_id, {}) if user_id else state["global"]
        return dict(scope)

    def clear(self, skill_name: str) -> None:
        state_file = self._get_state_file(skill_name)
        if state_file.exists():
            state_file.unlink()
