from __future__ import annotations

import json
from pathlib import Path

from ash.security.vault import FileVault, VaultError


def _single_record_file(root: Path) -> Path:
    matches = list(root.rglob("*.json"))
    assert len(matches) == 1
    return matches[0]


def test_file_vault_round_trip_and_delete(tmp_path: Path) -> None:
    vault = FileVault(tmp_path / "vault")

    ref = vault.put_json(
        namespace="gog.credentials",
        key="user-1:gog.email:work",
        payload={"credential_key": "cred_123"},
    )
    loaded = vault.get_json(ref)
    assert loaded == {"credential_key": "cred_123"}
    assert vault.delete(ref) is True
    assert vault.get_json(ref) is None
    assert vault.delete(ref) is False


def test_file_vault_is_stable_for_same_namespace_and_key(tmp_path: Path) -> None:
    vault = FileVault(tmp_path / "vault")

    ref1 = vault.put_json(
        namespace="gog.credentials",
        key="user-1:gog.email:work",
        payload={"credential_key": "cred_123"},
    )
    ref2 = vault.put_json(
        namespace="gog.credentials",
        key="user-1:gog.email:work",
        payload={"credential_key": "cred_456"},
    )
    assert ref1 == ref2
    assert vault.get_json(ref1) == {"credential_key": "cred_456"}


def test_file_vault_prunes_expired_records(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    vault = FileVault(root)

    ref = vault.put_json(
        namespace="gog.credentials",
        key="user-1:gog.email:work",
        payload={"credential_key": "cred_123"},
        ttl_seconds=60,
    )
    record_path = _single_record_file(root)
    raw = json.loads(record_path.read_text(encoding="utf-8"))
    raw["expires_at"] = 1
    record_path.write_text(json.dumps(raw, ensure_ascii=True), encoding="utf-8")

    assert vault.get_json(ref) is None
    assert not record_path.exists()


def test_file_vault_rejects_invalid_ref(tmp_path: Path) -> None:
    vault = FileVault(tmp_path / "vault")

    try:
        vault.get_json("not-a-ref")
    except VaultError:
        pass
    else:
        raise AssertionError("expected VaultError for invalid ref")
