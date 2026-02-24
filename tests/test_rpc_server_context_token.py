from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ash.context_token import ContextTokenService
from ash.rpc.server import RPCServer


def _service() -> ContextTokenService:
    return ContextTokenService(secret=b"test-secret-key-32-bytes-minimum")


@pytest.mark.asyncio
async def test_rpc_server_requires_context_token(tmp_path: Path) -> None:
    server = RPCServer(tmp_path / "rpc.sock", context_token_service=_service())

    async def _handler(params: dict[str, Any]) -> dict[str, Any]:
        return params

    server.register("echo", _handler)

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "echo",
        "params": {},
    }
    response = await server._process_request(json.dumps(payload).encode("utf-8"))

    assert response.error is not None
    assert response.error.code == -32602
    assert "context token" in response.error.message.lower()


@pytest.mark.asyncio
async def test_rpc_server_uses_verified_identity_claims(tmp_path: Path) -> None:
    service = _service()
    server = RPCServer(tmp_path / "rpc.sock", context_token_service=service)

    async def _handler(params: dict[str, Any]) -> dict[str, Any]:
        return params

    server.register("echo", _handler)

    token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
        session_key="telegram_chat-1_user-1",
        thread_id="thread-1",
        source_username="alice",
        source_display_name="Alice",
    )

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "echo",
        "params": {
            "context_token": token,
            "user_id": "attacker",
            "chat_id": "other-chat",
            "provider": "spoofed",
        },
    }
    response = await server._process_request(json.dumps(payload).encode("utf-8"))

    assert response.error is None
    assert response.result is not None
    assert response.result["user_id"] == "user-1"
    assert response.result["chat_id"] == "chat-1"
    assert response.result["provider"] == "telegram"
    assert response.result["thread_id"] == "thread-1"
    assert response.result["source_username"] == "alice"
    assert response.result["source_display_name"] == "Alice"


@pytest.mark.asyncio
async def test_rpc_server_keeps_browser_provider_param(tmp_path: Path) -> None:
    service = _service()
    server = RPCServer(tmp_path / "rpc.sock", context_token_service=service)

    async def _handler(params: dict[str, Any]) -> dict[str, Any]:
        return params

    server.register("browser.session.list", _handler)

    token = service.issue(
        effective_user_id="user-1",
        provider="telegram",
    )
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "browser.session.list",
        "params": {
            "context_token": token,
            "provider": "sandbox",
        },
    }
    response = await server._process_request(json.dumps(payload).encode("utf-8"))

    assert response.error is None
    assert response.result is not None
    assert response.result["user_id"] == "user-1"
    assert response.result["provider"] == "sandbox"
