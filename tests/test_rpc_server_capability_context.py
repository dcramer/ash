from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ash.capabilities import (
    CapabilityAuthBeginResult,
    CapabilityAuthCompleteResult,
    CapabilityCallContext,
    CapabilityDefinition,
    CapabilityManager,
)
from ash.capabilities.types import CapabilityOperation
from ash.context_token import ContextTokenService
from ash.rpc.methods.capability import register_capability_methods
from ash.rpc.server import RPCServer


def _service() -> ContextTokenService:
    return ContextTokenService(secret=b"test-secret-key-32-bytes-minimum")


class _ContextCaptureProvider:
    namespace = "gog"

    def __init__(self) -> None:
        self.invoke_context: CapabilityCallContext | None = None

    async def definitions(self) -> list[CapabilityDefinition]:
        return [
            CapabilityDefinition(
                id="gog.email",
                description="Email ops",
                sensitive=True,
                operations={
                    "list_messages": CapabilityOperation(
                        name="list_messages",
                        description="List inbox",
                        requires_auth=False,
                    )
                },
            )
        ]

    async def auth_begin(
        self,
        *,
        capability_id: str,
        account_hint: str | None,
        context: CapabilityCallContext,
    ) -> CapabilityAuthBeginResult:
        _ = capability_id
        _ = account_hint
        _ = context
        return CapabilityAuthBeginResult(auth_url="https://auth.example/gog.email")

    async def auth_complete(
        self,
        *,
        capability_id: str,
        flow_state: dict[str, Any],
        callback_url: str | None,
        code: str | None,
        context: CapabilityCallContext,
    ) -> CapabilityAuthCompleteResult:
        _ = capability_id
        _ = flow_state
        _ = callback_url
        _ = code
        _ = context
        return CapabilityAuthCompleteResult(account_ref="work")

    async def invoke(
        self,
        *,
        capability_id: str,
        operation: str,
        input_data: dict[str, Any],
        account_ref: str | None,
        idempotency_key: str | None,
        context: CapabilityCallContext,
    ) -> dict[str, Any]:
        _ = capability_id
        _ = operation
        _ = input_data
        _ = account_ref
        _ = idempotency_key
        self.invoke_context = context
        return {"status": "ok"}


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


@pytest.mark.asyncio
async def test_capability_rpc_projects_trusted_context_for_provider_calls(
    tmp_path: Path,
) -> None:
    service = _service()
    manager = CapabilityManager(auth_flow_ttl_seconds=300)
    provider = _ContextCaptureProvider()
    await manager.register_provider(provider)
    server = RPCServer(tmp_path / "rpc.sock", context_token_service=service)
    register_capability_methods(server, manager)

    token = service.issue(
        effective_user_id="verified-user",
        chat_id="chat-from-token",
        chat_type="private",
        provider="telegram",
        session_key="session-from-token",
        thread_id="thread-from-token",
        source_username="verified_username",
        source_display_name="Verified Name",
    )
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "capability.invoke",
        "params": {
            "context_token": token,
            "capability": "gog.email",
            "operation": "list_messages",
            "input": {"folder": "inbox"},
            # Spoofed values should be overwritten by token projection.
            "user_id": "spoof-user",
            "chat_id": "spoof-chat",
            "provider": "spoof-provider",
            "session_key": "spoof-session",
            "thread_id": "spoof-thread",
            "source_username": "spoof-name",
            "source_display_name": "Spoof Display",
        },
    }
    response = await server._process_request(json.dumps(payload).encode("utf-8"))
    assert response.error is None

    assert provider.invoke_context is not None
    assert provider.invoke_context.user_id == "verified-user"
    assert provider.invoke_context.chat_id == "chat-from-token"
    assert provider.invoke_context.provider == "telegram"
    assert provider.invoke_context.session_key == "session-from-token"
    assert provider.invoke_context.thread_id == "thread-from-token"
    assert provider.invoke_context.source_username == "verified_username"
    assert provider.invoke_context.source_display_name == "Verified Name"
