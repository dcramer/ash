"""Host-side vault abstraction for sensitive credential material."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol

from filelock import FileLock

from ash.config.paths import get_vault_path

_VAULT_VERSION = 1
_REF_PREFIX = f"vault:v{_VAULT_VERSION}"
_NAMESPACE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_RECORD_ID_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class VaultError(ValueError):
    """Vault operation error."""


class Vault(Protocol):
    """Interface for storing/retrieving sensitive JSON payloads."""

    def put_json(
        self,
        *,
        namespace: str,
        key: str,
        payload: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> str: ...

    def get_json(self, ref: str) -> dict[str, Any] | None: ...

    def delete(self, ref: str) -> bool: ...


class FileVault(Vault):
    """File-backed vault with strict file permissions."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or get_vault_path()
        self._lock = FileLock(str(self._root) + ".lock")

    def put_json(
        self,
        *,
        namespace: str,
        key: str,
        payload: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> str:
        normalized_namespace = _normalize_namespace(namespace)
        normalized_key = _normalize_key(key)
        record_id = _record_id(normalized_namespace, normalized_key)
        now_epoch = int(time.time())
        expires_at = (
            now_epoch + max(1, int(ttl_seconds)) if ttl_seconds is not None else None
        )
        record = {
            "version": _VAULT_VERSION,
            "namespace": normalized_namespace,
            "key_hash": hashlib.sha256(normalized_key.encode("utf-8")).hexdigest(),
            "created_at": now_epoch,
            "expires_at": expires_at,
            "payload": dict(payload),
        }
        with self._lock:
            self._write_record(normalized_namespace, record_id, record)
        return _build_ref(normalized_namespace, record_id)

    def get_json(self, ref: str) -> dict[str, Any] | None:
        namespace, record_id = _parse_ref(ref)
        with self._lock:
            record = self._read_record(namespace, record_id)
            if record is None:
                return None
            now_epoch = int(time.time())
            expires_at = _optional_int(record.get("expires_at"))
            if expires_at is not None and expires_at <= now_epoch:
                self._delete_record(namespace, record_id)
                return None
            payload = record.get("payload")
            if not isinstance(payload, dict):
                return None
            return dict(payload)

    def delete(self, ref: str) -> bool:
        namespace, record_id = _parse_ref(ref)
        with self._lock:
            return self._delete_record(namespace, record_id)

    def _namespace_dir(self, namespace: str) -> Path:
        return self._root / namespace

    def _record_path(self, namespace: str, record_id: str) -> Path:
        return self._namespace_dir(namespace) / f"{record_id}.json"

    def _write_record(
        self,
        namespace: str,
        record_id: str,
        record: dict[str, Any],
    ) -> None:
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._root.chmod(0o700)
        namespace_dir = self._namespace_dir(namespace)
        namespace_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        namespace_dir.chmod(0o700)
        target = self._record_path(namespace, record_id)
        with NamedTemporaryFile(
            mode="w",
            delete=False,
            dir=str(namespace_dir),
            encoding="utf-8",
        ) as handle:
            json.dump(record, handle, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            temp_path = Path(handle.name)
        temp_path.chmod(0o600)
        temp_path.replace(target)
        target.chmod(0o600)

    def _read_record(self, namespace: str, record_id: str) -> dict[str, Any] | None:
        path = self._record_path(namespace, record_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        if _optional_int(raw.get("version")) != _VAULT_VERSION:
            return None
        stored_namespace = raw.get("namespace")
        if stored_namespace != namespace:
            return None
        return raw

    def _delete_record(self, namespace: str, record_id: str) -> bool:
        path = self._record_path(namespace, record_id)
        if not path.exists():
            return False
        path.unlink()
        return True


def _normalize_namespace(namespace: str) -> str:
    text = str(namespace).strip().lower()
    if not text or not _NAMESPACE_PATTERN.match(text):
        raise VaultError("invalid vault namespace")
    return text


def _normalize_key(key: str) -> str:
    text = str(key).strip()
    if not text:
        raise VaultError("invalid vault key")
    return text


def _record_id(namespace: str, key: str) -> str:
    digest = hashlib.sha256(f"{namespace}:{key}".encode()).hexdigest()
    return digest


def _build_ref(namespace: str, record_id: str) -> str:
    return f"{_REF_PREFIX}:{namespace}:{record_id}"


def _parse_ref(ref: str) -> tuple[str, str]:
    text = str(ref).strip()
    parts = text.split(":")
    if len(parts) != 4:
        raise VaultError("invalid vault ref format")
    prefix = ":".join(parts[:2])
    if prefix != _REF_PREFIX:
        raise VaultError("unsupported vault ref version")
    namespace = _normalize_namespace(parts[2])
    record_id = parts[3].strip().lower()
    if not _RECORD_ID_PATTERN.match(record_id):
        raise VaultError("invalid vault record id")
    return namespace, record_id


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return None
