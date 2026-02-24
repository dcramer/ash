from __future__ import annotations

import json
from pathlib import Path

import pytest

from ash.capabilities import CapabilityDefinition, CapabilityManager
from ash.capabilities.types import CapabilityOperation
from ash.context_token import ContextTokenService
from ash.rpc.methods.capability import register_capability_methods
from ash.rpc.server import RPCServer


def _service() -> ContextTokenService:
    return ContextTokenService(secret=b"test-secret-key-32-bytes-minimum")


@pytest.mark.asyncio
async def test_capability_rpc_uses_verified_user_scope(tmp_path: Path) -> None:
    service = _service()
    manager = CapabilityManager(auth_flow_ttl_seconds=300)
    await manager.register(
        CapabilityDefinition(
            id="gog.email",
            description="Email ops",
            sensitive=True,
            operations={
                "list_messages": CapabilityOperation(
                    name="list_messages",
                    description="List inbox",
                    requires_auth=True,
                )
            },
        )
    )

    server = RPCServer(tmp_path / "rpc.sock", context_token_service=service)
    register_capability_methods(server, manager)

    owner_token = service.issue(
        effective_user_id="user-1",
        chat_type="private",
        provider="telegram",
    )
    begin_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "capability.auth.begin",
        "params": {
            "context_token": owner_token,
            "capability": "gog.email",
            "account_hint": "work",
        },
    }
    begin_response = await server._process_request(
        json.dumps(begin_payload).encode("utf-8")
    )
    assert begin_response.error is None
    assert begin_response.result is not None
    flow_id = str(begin_response.result["flow_id"])

    complete_payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "capability.auth.complete",
        "params": {
            "context_token": owner_token,
            "flow_id": flow_id,
            "callback_url": "https://localhost/callback?code=abc",
        },
    }
    complete_response = await server._process_request(
        json.dumps(complete_payload).encode("utf-8")
    )
    assert complete_response.error is None
    assert complete_response.result is not None
    assert complete_response.result["ok"] is True

    attacker_token = service.issue(
        effective_user_id="user-2",
        chat_type="private",
        provider="telegram",
    )
    invoke_payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "capability.invoke",
        "params": {
            "context_token": attacker_token,
            "capability": "gog.email",
            "operation": "list_messages",
            "input": {"folder": "inbox"},
            # Attempted spoof should be ignored by server projection.
            "user_id": "user-1",
        },
    }
    invoke_response = await server._process_request(
        json.dumps(invoke_payload).encode("utf-8")
    )
    assert invoke_response.error is not None
    assert "capability_auth_required" in invoke_response.error.message


@pytest.mark.asyncio
async def test_capability_rpc_rejects_unqualified_capability_ids(
    tmp_path: Path,
) -> None:
    service = _service()
    manager = CapabilityManager(auth_flow_ttl_seconds=300)
    server = RPCServer(tmp_path / "rpc.sock", context_token_service=service)
    register_capability_methods(server, manager)
    token = service.issue(
        effective_user_id="user-1",
        chat_type="private",
    )

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "capability.invoke",
        "params": {
            "context_token": token,
            "capability": "email",
            "operation": "list_messages",
            "input": {},
        },
    }
    response = await server._process_request(json.dumps(payload).encode("utf-8"))
    assert response.error is not None
    assert "capability_invalid_input" in response.error.message
