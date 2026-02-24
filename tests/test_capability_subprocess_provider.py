from __future__ import annotations

from typing import Any

import pytest

from ash.capabilities import CapabilityError
from ash.capabilities.providers import (
    CapabilityCallContext,
    SubprocessCapabilityProvider,
)


def _context() -> CapabilityCallContext:
    return CapabilityCallContext(
        user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
        thread_id="thread-1",
        session_key="session-1",
        source_username="alice",
        source_display_name="Alice",
    )


@pytest.mark.asyncio
async def test_subprocess_provider_parses_definitions(monkeypatch) -> None:
    async def _fake_execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["method"] == "definitions"
        assert payload["version"] == 1
        return {
            "version": 1,
            "id": payload["id"],
            "result": {
                "definitions": [
                    {
                        "id": "gog.email",
                        "description": "Email",
                        "sensitive": True,
                        "operations": [
                            {
                                "name": "list_messages",
                                "description": "List inbox",
                                "requires_auth": True,
                            }
                        ],
                    }
                ]
            },
        }

    monkeypatch.setattr(
        SubprocessCapabilityProvider,
        "_execute_command",
        _fake_execute,
    )
    provider = SubprocessCapabilityProvider(namespace="gog", command=["gogcli", "rpc"])

    definitions = await provider.definitions()
    assert len(definitions) == 1
    assert definitions[0].id == "gog.email"
    assert "list_messages" in definitions[0].operations


@pytest.mark.asyncio
async def test_subprocess_provider_auth_and_invoke(monkeypatch) -> None:
    async def _fake_execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        method = payload["method"]
        if method == "auth_begin":
            return {
                "version": 1,
                "id": payload["id"],
                "result": {
                    "auth_url": "https://example.test/auth",
                    "flow_state": {"nonce": "n1"},
                },
            }
        if method == "auth_complete":
            return {
                "version": 1,
                "id": payload["id"],
                "result": {
                    "account_ref": "work",
                    "credential_material": {"credential_key": "cred_123"},
                },
            }
        if method == "invoke":
            return {
                "version": 1,
                "id": payload["id"],
                "result": {"output": {"status": "ok", "messages": []}},
            }
        raise AssertionError(f"unexpected method: {method}")

    monkeypatch.setattr(
        SubprocessCapabilityProvider,
        "_execute_command",
        _fake_execute,
    )
    provider = SubprocessCapabilityProvider(namespace="gog", command="gogcli rpc")

    begin = await provider.auth_begin(
        capability_id="gog.email",
        account_hint="work",
        context=_context(),
    )
    assert begin.auth_url == "https://example.test/auth"
    assert begin.flow_state == {"nonce": "n1"}

    complete = await provider.auth_complete(
        capability_id="gog.email",
        flow_state=begin.flow_state,
        callback_url="https://localhost/callback?code=abc",
        code=None,
        context=_context(),
    )
    assert complete.account_ref == "work"
    assert complete.credential_material == {"credential_key": "cred_123"}

    output = await provider.invoke(
        capability_id="gog.email",
        operation="list_messages",
        input_data={"folder": "inbox"},
        account_ref="work",
        idempotency_key="idem-1",
        context=_context(),
    )
    assert output == {"status": "ok", "messages": []}


@pytest.mark.asyncio
async def test_subprocess_provider_surfaces_bridge_errors(monkeypatch) -> None:
    async def _fake_execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["version"] == 1
        return {
            "version": 1,
            "id": payload["id"],
            "error": {
                "code": "capability_backend_unavailable",
                "message": "bridge offline",
            },
        }

    monkeypatch.setattr(
        SubprocessCapabilityProvider,
        "_execute_command",
        _fake_execute,
    )
    provider = SubprocessCapabilityProvider(namespace="gog", command=["gogcli", "rpc"])

    with pytest.raises(CapabilityError) as exc_info:
        await provider.definitions()
    assert exc_info.value.code == "capability_backend_unavailable"


@pytest.mark.asyncio
async def test_subprocess_provider_rejects_response_version_mismatch(
    monkeypatch,
) -> None:
    async def _fake_execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "version": 999,
            "id": payload["id"],
            "result": {"definitions": []},
        }

    monkeypatch.setattr(
        SubprocessCapabilityProvider,
        "_execute_command",
        _fake_execute,
    )
    provider = SubprocessCapabilityProvider(namespace="gog", command=["gogcli", "rpc"])

    with pytest.raises(CapabilityError) as exc_info:
        await provider.definitions()
    assert exc_info.value.code == "capability_invalid_output"


@pytest.mark.asyncio
async def test_subprocess_provider_rejects_response_id_mismatch(monkeypatch) -> None:
    async def _fake_execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "version": 1,
            "id": "wrong-id",
            "result": {"definitions": []},
        }

    monkeypatch.setattr(
        SubprocessCapabilityProvider,
        "_execute_command",
        _fake_execute,
    )
    provider = SubprocessCapabilityProvider(namespace="gog", command=["gogcli", "rpc"])

    with pytest.raises(CapabilityError) as exc_info:
        await provider.definitions()
    assert exc_info.value.code == "capability_invalid_output"


@pytest.mark.asyncio
async def test_subprocess_provider_rejects_result_error_ambiguity(monkeypatch) -> None:
    async def _fake_execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "version": 1,
            "id": payload["id"],
            "result": {"definitions": []},
            "error": {
                "code": "capability_backend_unavailable",
                "message": "bridge offline",
            },
        }

    monkeypatch.setattr(
        SubprocessCapabilityProvider,
        "_execute_command",
        _fake_execute,
    )
    provider = SubprocessCapabilityProvider(namespace="gog", command=["gogcli", "rpc"])

    with pytest.raises(CapabilityError) as exc_info:
        await provider.definitions()
    assert exc_info.value.code == "capability_invalid_output"


@pytest.mark.asyncio
async def test_subprocess_provider_rejects_bridge_error_missing_code(
    monkeypatch,
) -> None:
    async def _fake_execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "version": 1,
            "id": payload["id"],
            "error": {
                "message": "bridge offline",
            },
        }

    monkeypatch.setattr(
        SubprocessCapabilityProvider,
        "_execute_command",
        _fake_execute,
    )
    provider = SubprocessCapabilityProvider(namespace="gog", command=["gogcli", "rpc"])

    with pytest.raises(CapabilityError) as exc_info:
        await provider.definitions()
    assert exc_info.value.code == "capability_invalid_output"


@pytest.mark.asyncio
async def test_subprocess_provider_rejects_result_error_absence(monkeypatch) -> None:
    async def _fake_execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "version": 1,
            "id": payload["id"],
        }

    monkeypatch.setattr(
        SubprocessCapabilityProvider,
        "_execute_command",
        _fake_execute,
    )
    provider = SubprocessCapabilityProvider(namespace="gog", command=["gogcli", "rpc"])

    with pytest.raises(CapabilityError) as exc_info:
        await provider.definitions()
    assert exc_info.value.code == "capability_invalid_output"


@pytest.mark.asyncio
async def test_subprocess_provider_rejects_result_non_object(monkeypatch) -> None:
    async def _fake_execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "version": 1,
            "id": payload["id"],
            "result": [],
        }

    monkeypatch.setattr(
        SubprocessCapabilityProvider,
        "_execute_command",
        _fake_execute,
    )
    provider = SubprocessCapabilityProvider(namespace="gog", command=["gogcli", "rpc"])

    with pytest.raises(CapabilityError) as exc_info:
        await provider.definitions()
    assert exc_info.value.code == "capability_invalid_output"
