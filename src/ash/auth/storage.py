"""Credential storage for OAuth providers."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from filelock import FileLock

from ash.config.paths import get_ash_home

logger = logging.getLogger(__name__)


@dataclass
class OAuthCredentials:
    """OAuth token credentials."""

    access: str
    refresh: str
    expires: float  # Unix timestamp in seconds
    account_id: str


class AuthStorage:
    """Read/write OAuth credentials from ~/.ash/auth.json.

    Uses file locking for safe concurrent access. File permissions
    are set to 0o600 (owner-only read/write).
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (get_ash_home() / "auth.json")
        self._lock = FileLock(str(self._path) + ".lock")

    @property
    def path(self) -> Path:
        return self._path

    def _read_all(self) -> dict[str, dict[str, str | float]]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read auth.json: %s", e)
            return {}

    def _write_all(self, data: dict[str, dict[str, str | float]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2) + "\n")
        # Set restrictive permissions (owner only)
        self._path.chmod(0o600)

    def load(self, provider_id: str) -> OAuthCredentials | None:
        """Load credentials for a provider.

        Returns:
            OAuthCredentials or None if not found.
        """
        with self._lock:
            data = self._read_all()
        entry = data.get(provider_id)
        if not entry:
            return None
        try:
            return OAuthCredentials(
                access=str(entry["access"]),
                refresh=str(entry["refresh"]),
                expires=float(entry["expires"]),
                account_id=str(entry["account_id"]),
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Invalid credentials for %s: %s", provider_id, e)
            return None

    def save(self, provider_id: str, credentials: OAuthCredentials) -> None:
        """Save credentials for a provider."""
        with self._lock:
            data = self._read_all()
            data[provider_id] = asdict(credentials)
            self._write_all(data)

    def remove(self, provider_id: str) -> bool:
        """Remove credentials for a provider.

        Returns:
            True if credentials were removed, False if not found.
        """
        with self._lock:
            data = self._read_all()
            if provider_id not in data:
                return False
            del data[provider_id]
            self._write_all(data)
        return True

    def list_providers(self) -> list[str]:
        """List all providers with stored credentials."""
        with self._lock:
            data = self._read_all()
        return list(data.keys())
