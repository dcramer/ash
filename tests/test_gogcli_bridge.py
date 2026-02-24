from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ash.capabilities import CapabilityError
from ash.capabilities.providers import (
    CapabilityCallContext,
    SubprocessCapabilityProvider,
)
from ash.context_token import ContextTokenService

_BRIDGE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gogcli_bridge.py"


def _load_state(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _run_bridge(
    payload: dict[str, Any],
    *,
    env: dict[str, str],
) -> dict[str, Any]:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(_BRIDGE_SCRIPT), "bridge"],  # noqa: S607
        input=json.dumps(payload, ensure_ascii=True),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, dict)
    return parsed


def _context(user_id: str) -> CapabilityCallContext:
    return CapabilityCallContext(
        user_id=user_id,
        chat_id=f"chat-{user_id}",
        chat_type="private",
        provider="telegram",
        thread_id=f"thread-{user_id}",
        session_key=f"session-{user_id}",
        source_username=user_id,
        source_display_name=user_id.title(),
    )


def test_bridge_definitions() -> None:
    response = _run_bridge(
        {
            "version": 1,
            "id": "req_definitions",
            "namespace": "gog",
            "method": "definitions",
            "params": {},
        },
        env={},
    )
    assert response["version"] == 1
    assert response["id"] == "req_definitions"
    assert "result" in response

    result = response["result"]
    assert isinstance(result, dict)
    definitions = result["definitions"]
    assert isinstance(definitions, list)
    ids = {
        item["id"] for item in definitions if isinstance(item, dict) and "id" in item
    }
    assert ids == {"gog.email", "gog.calendar"}


def test_bridge_auth_flow_and_user_scoped_invoke(tmp_path: Path) -> None:
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    state_path = tmp_path / "gogcli-state.json"
    env = {
        "ASH_CONTEXT_TOKEN_SECRET": service.export_verifier_secret(),
        "GOGCLI_STATE_PATH": str(state_path),
    }
    user1_token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
    )

    begin = _run_bridge(
        {
            "version": 1,
            "id": "req_auth_begin",
            "namespace": "gog",
            "method": "auth_begin",
            "params": {
                "capability_id": "gog.email",
                "account_hint": "work",
                "context_token": user1_token,
            },
        },
        env=env,
    )
    assert "error" not in begin
    flow_state = begin["result"]["flow_state"]
    assert flow_state["flow_id"]
    assert flow_state["nonce"]

    state_after_begin = _load_state(state_path)
    assert flow_state["flow_id"] in state_after_begin["auth_flows"]

    complete = _run_bridge(
        {
            "version": 1,
            "id": "req_auth_complete",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "code": "sample-code",
                "context_token": user1_token,
            },
        },
        env=env,
    )
    assert "error" not in complete
    account_ref = complete["result"]["account_ref"]
    assert account_ref == "work"
    state_after_complete = _load_state(state_path)
    assert flow_state["flow_id"] not in state_after_complete["auth_flows"]

    invoke_user1 = _run_bridge(
        {
            "version": 1,
            "id": "req_invoke_user1",
            "namespace": "gog",
            "method": "invoke",
            "params": {
                "capability_id": "gog.email",
                "operation": "list_messages",
                "input_data": {"folder": "inbox", "limit": 2},
                "account_ref": account_ref,
                "context_token": user1_token,
            },
        },
        env=env,
    )
    assert "error" not in invoke_user1
    messages = invoke_user1["result"]["output"]["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 2
    state_after_invoke = _load_state(state_path)
    account_key = "user-1:gog.email:work"
    assert account_key in state_after_invoke["accounts"]
    scope_key = "user-1:gog.email"
    assert state_after_invoke["operation_state"][scope_key]["invoke_count"] == 1

    user2_token = service.issue(
        effective_user_id="user-2",
        chat_id="chat-2",
        chat_type="private",
        provider="telegram",
    )
    invoke_user2 = _run_bridge(
        {
            "version": 1,
            "id": "req_invoke_user2",
            "namespace": "gog",
            "method": "invoke",
            "params": {
                "capability_id": "gog.email",
                "operation": "list_messages",
                "input_data": {"folder": "inbox", "limit": 2},
                "account_ref": account_ref,
                "context_token": user2_token,
            },
        },
        env=env,
    )
    assert invoke_user2["error"]["code"] == "capability_auth_required"


def test_bridge_auth_complete_rejects_reused_flow_state(tmp_path: Path) -> None:
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    env = {
        "ASH_CONTEXT_TOKEN_SECRET": service.export_verifier_secret(),
        "GOGCLI_STATE_PATH": str(tmp_path / "gogcli-state.json"),
    }
    user_token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
    )

    begin = _run_bridge(
        {
            "version": 1,
            "id": "req_reuse_begin",
            "namespace": "gog",
            "method": "auth_begin",
            "params": {
                "capability_id": "gog.email",
                "account_hint": "work",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in begin
    flow_state = begin["result"]["flow_state"]

    first_complete = _run_bridge(
        {
            "version": 1,
            "id": "req_reuse_complete_1",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "code": "sample-code",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in first_complete

    second_complete = _run_bridge(
        {
            "version": 1,
            "id": "req_reuse_complete_2",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "code": "sample-code",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert second_complete["error"]["code"] == "capability_auth_flow_invalid"


def test_bridge_auth_complete_rejects_expired_flow_state(tmp_path: Path) -> None:
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    state_path = tmp_path / "gogcli-state.json"
    env = {
        "ASH_CONTEXT_TOKEN_SECRET": service.export_verifier_secret(),
        "GOGCLI_STATE_PATH": str(state_path),
    }
    user_token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
    )

    begin = _run_bridge(
        {
            "version": 1,
            "id": "req_expired_begin",
            "namespace": "gog",
            "method": "auth_begin",
            "params": {
                "capability_id": "gog.email",
                "account_hint": "work",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in begin
    flow_state = begin["result"]["flow_state"]
    flow_id = flow_state["flow_id"]

    state = _load_state(state_path)
    state["auth_flows"][flow_id]["expires_at"] = 1
    state_path.write_text(json.dumps(state, ensure_ascii=True), encoding="utf-8")

    complete = _run_bridge(
        {
            "version": 1,
            "id": "req_expired_complete",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "code": "sample-code",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert complete["error"]["code"] == "capability_auth_flow_invalid"


def test_bridge_rejects_invalid_context_signature(tmp_path: Path) -> None:
    signer = ContextTokenService(secret=b"bridge-signing-secret-32-bytes...")
    verifier = ContextTokenService(secret=b"bridge-verifier-secret-32-bytes..")
    env = {
        "ASH_CONTEXT_TOKEN_SECRET": verifier.export_verifier_secret(),
        "GOGCLI_STATE_PATH": str(tmp_path / "gogcli-state.json"),
    }
    bad_token = signer.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
    )

    response = _run_bridge(
        {
            "version": 1,
            "id": "req_bad_sig",
            "namespace": "gog",
            "method": "auth_begin",
            "params": {
                "capability_id": "gog.email",
                "context_token": bad_token,
            },
        },
        env=env,
    )
    assert response["error"]["code"] == "capability_invalid_input"


@pytest.mark.asyncio
async def test_subprocess_provider_round_trip_with_bridge(tmp_path: Path) -> None:
    service = ContextTokenService(secret=b"provider-roundtrip-secret-32-bytes")
    env = {
        "GOGCLI_STATE_PATH": str(tmp_path / "gogcli-state.json"),
    }

    with pytest.MonkeyPatch.context() as mp:
        for key, value in env.items():
            mp.setenv(key, value)

        provider = SubprocessCapabilityProvider(
            namespace="gog",
            command=[sys.executable, str(_BRIDGE_SCRIPT), "bridge"],
            context_token_service=service,
        )
        definitions = await provider.definitions()
        ids = {definition.id for definition in definitions}
        assert ids == {"gog.email", "gog.calendar"}

        user1 = _context("user-1")
        begin = await provider.auth_begin(
            capability_id="gog.email",
            account_hint="work",
            context=user1,
        )
        complete = await provider.auth_complete(
            capability_id="gog.email",
            flow_state=begin.flow_state,
            callback_url=None,
            code="sample",
            context=user1,
        )
        output = await provider.invoke(
            capability_id="gog.email",
            operation="list_messages",
            input_data={"limit": 1},
            account_ref=complete.account_ref,
            idempotency_key=None,
            context=user1,
        )
        assert output["count"] == 1

        user2 = _context("user-2")
        with pytest.raises(CapabilityError) as exc_info:
            await provider.invoke(
                capability_id="gog.email",
                operation="list_messages",
                input_data={"limit": 1},
                account_ref=complete.account_ref,
                idempotency_key=None,
                context=user2,
            )
        assert exc_info.value.code == "capability_auth_required"
