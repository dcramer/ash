"""JSONL-backed storage for browser profiles and sessions."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from ash.browser.types import (
    BrowserProfile,
    BrowserProfileStatus,
    BrowserProviderName,
    BrowserSession,
    BrowserSessionStatus,
)


class BrowserStore:
    """Append-only JSONL browser store with replay-based reads."""

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        self._sessions_path = base_dir / "sessions.jsonl"
        self._profiles_path = base_dir / "profiles.jsonl"
        self._artifacts_dir = base_dir / "artifacts"

    @property
    def artifacts_dir(self) -> Path:
        return self._artifacts_dir

    def ensure_dirs(self) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _provider_name(value: object) -> BrowserProviderName:
        if value == "kernel":
            return "kernel"
        return "sandbox"

    @staticmethod
    def _session_status(value: object) -> BrowserSessionStatus:
        if value in {"active", "closed", "archived"}:
            return cast("BrowserSessionStatus", value)
        return "active"

    @staticmethod
    def _profile_status(value: object) -> BrowserProfileStatus:
        if value in {"active", "archived"}:
            return cast("BrowserProfileStatus", value)
        return "active"

    def append_session(self, session: BrowserSession) -> None:
        self.ensure_dirs()
        payload = asdict(session)
        payload["created_at"] = session.created_at.isoformat()
        payload["updated_at"] = session.updated_at.isoformat()
        self._append_line(self._sessions_path, payload)

    def append_profile(self, profile: BrowserProfile) -> None:
        self.ensure_dirs()
        payload = asdict(profile)
        payload["created_at"] = profile.created_at.isoformat()
        payload["updated_at"] = profile.updated_at.isoformat()
        self._append_line(self._profiles_path, payload)

    def list_sessions(
        self,
        *,
        effective_user_id: str | None = None,
        include_archived: bool = False,
    ) -> list[BrowserSession]:
        sessions = self._load_latest_sessions()
        rows = list(sessions.values())
        if effective_user_id is not None:
            rows = [s for s in rows if s.effective_user_id == effective_user_id]
        if not include_archived:
            rows = [s for s in rows if s.status != "archived"]
        rows.sort(key=lambda s: s.updated_at, reverse=True)
        return rows

    def get_session(self, session_id: str) -> BrowserSession | None:
        return self._load_latest_sessions().get(session_id)

    def get_session_by_name(
        self,
        *,
        name: str,
        effective_user_id: str,
        provider: str,
        include_archived: bool = False,
    ) -> BrowserSession | None:
        candidates = [
            s
            for s in self._load_latest_sessions().values()
            if s.name == name
            and s.effective_user_id == effective_user_id
            and s.provider == provider
        ]
        if not include_archived:
            candidates = [s for s in candidates if s.status != "archived"]
        if not candidates:
            return None
        candidates.sort(key=lambda s: s.updated_at, reverse=True)
        return candidates[0]

    def list_profiles(
        self,
        *,
        effective_user_id: str | None = None,
        include_archived: bool = False,
    ) -> list[BrowserProfile]:
        profiles = list(self._load_latest_profiles().values())
        if effective_user_id is not None:
            profiles = [p for p in profiles if p.effective_user_id == effective_user_id]
        if not include_archived:
            profiles = [p for p in profiles if p.status != "archived"]
        profiles.sort(key=lambda p: p.updated_at, reverse=True)
        return profiles

    def get_profile(
        self,
        *,
        name: str,
        effective_user_id: str,
        provider: str,
    ) -> BrowserProfile | None:
        key = self._profile_key(name, effective_user_id, provider)
        return self._load_latest_profiles().get(key)

    def _profile_key(self, name: str, effective_user_id: str, provider: str) -> str:
        return f"{effective_user_id}:{provider}:{name}"

    def _load_latest_sessions(self) -> dict[str, BrowserSession]:
        latest: dict[str, BrowserSession] = {}
        for payload in self._iter_jsonl(self._sessions_path):
            try:
                session = BrowserSession(
                    id=str(payload["id"]),
                    name=str(payload["name"]),
                    effective_user_id=str(payload["effective_user_id"]),
                    provider=self._provider_name(payload["provider"]),
                    status=self._session_status(payload.get("status", "active")),
                    profile_name=payload.get("profile_name"),
                    provider_session_id=payload.get("provider_session_id"),
                    current_url=payload.get("current_url"),
                    last_error=payload.get("last_error"),
                    metadata=dict(payload.get("metadata") or {}),
                    created_at=self._parse_dt(payload.get("created_at")),
                    updated_at=self._parse_dt(payload.get("updated_at")),
                )
            except (KeyError, TypeError, ValueError):
                continue
            latest[session.id] = session
        return latest

    def _load_latest_profiles(self) -> dict[str, BrowserProfile]:
        latest: dict[str, BrowserProfile] = {}
        for payload in self._iter_jsonl(self._profiles_path):
            try:
                profile = BrowserProfile(
                    name=str(payload["name"]),
                    effective_user_id=str(payload["effective_user_id"]),
                    provider=self._provider_name(payload["provider"]),
                    status=self._profile_status(payload.get("status", "active")),
                    created_at=self._parse_dt(payload.get("created_at")),
                    updated_at=self._parse_dt(payload.get("updated_at")),
                )
            except (KeyError, TypeError, ValueError):
                continue
            latest[
                self._profile_key(
                    profile.name, profile.effective_user_id, profile.provider
                )
            ] = profile
        return latest

    def _iter_jsonl(self, path: Path):
        if not path.exists():
            return
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield payload

    def _append_line(self, path: Path, payload: dict) -> None:
        with path.open("a") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _parse_dt(self, value: object) -> datetime:
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now(UTC)
