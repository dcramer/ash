from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ash.capabilities import CapabilityManager
from ash.capabilities.providers import SubprocessCapabilityProvider
from ash.context_token import ContextTokenService
from ash.rpc.methods.capability import register_capability_methods
from ash.rpc.server import RPCServer
from ash.security.vault import FileVault

_BRIDGE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gogcli_bridge.py"


def _token(
    service: ContextTokenService,
    *,
    user_id: str,
    chat_type: str,
    chat_id: str,
) -> str:
    return service.issue(
        effective_user_id=user_id,
        chat_id=chat_id,
        chat_type=chat_type,
        provider="telegram",
        session_key=f"session-{user_id}-{chat_type}",
        thread_id=f"thread-{chat_id}",
        source_username=user_id,
        source_display_name=user_id.title(),
    )


async def _rpc(
    server: RPCServer,
    *,
    request_id: int,
    method: str,
    params: dict[str, object],
):
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }
    return await server._process_request(json.dumps(payload).encode("utf-8"))


@pytest.mark.asyncio
async def test_gog_capability_rpc_stack_round_trip_and_policy_enforcement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "gogcli-state.json"
    vault_path = tmp_path / "vault"
    monkeypatch.setenv("GOGCLI_STATE_PATH", str(state_path))
    monkeypatch.setenv("GOGCLI_VAULT_PATH", str(vault_path))

    service = ContextTokenService(secret=b"gog-capability-e2e-secret-32-bytes")
    manager = CapabilityManager(auth_flow_ttl_seconds=300)
    provider = SubprocessCapabilityProvider(
        namespace="gog",
        command=[sys.executable, str(_BRIDGE_SCRIPT), "bridge"],
        context_token_service=service,
    )
    await manager.register_provider(provider)

    server = RPCServer(tmp_path / "rpc.sock", context_token_service=service)
    register_capability_methods(server, manager)

    user1_private = _token(
        service,
        user_id="user-1",
        chat_type="private",
        chat_id="dm-user-1",
    )
    user1_group = _token(
        service,
        user_id="user-1",
        chat_type="group",
        chat_id="group-1",
    )
    user2_private = _token(
        service,
        user_id="user-2",
        chat_type="private",
        chat_id="dm-user-2",
    )

    list_private = await _rpc(
        server,
        request_id=1,
        method="capability.list",
        params={"context_token": user1_private},
    )
    assert list_private.error is None
    assert isinstance(list_private.result, dict)
    private_caps = {row["id"]: row for row in list_private.result["capabilities"]}
    assert set(private_caps) == {"gog.email", "gog.calendar"}
    assert private_caps["gog.email"]["available"] is True
    assert private_caps["gog.calendar"]["available"] is True
    assert private_caps["gog.email"]["authenticated"] is False
    assert private_caps["gog.calendar"]["authenticated"] is False

    list_group = await _rpc(
        server,
        request_id=2,
        method="capability.list",
        params={"context_token": user1_group, "include_unavailable": True},
    )
    assert list_group.error is None
    assert isinstance(list_group.result, dict)
    group_caps = {row["id"]: row for row in list_group.result["capabilities"]}
    assert group_caps["gog.email"]["available"] is False
    assert group_caps["gog.calendar"]["available"] is False

    begin_email = await _rpc(
        server,
        request_id=3,
        method="capability.auth.begin",
        params={
            "context_token": user1_private,
            "capability": "gog.email",
            "account_hint": "work",
        },
    )
    assert begin_email.error is None
    assert isinstance(begin_email.result, dict)
    email_flow_id = str(begin_email.result["flow_id"])

    complete_email = await _rpc(
        server,
        request_id=4,
        method="capability.auth.complete",
        params={
            "context_token": user1_private,
            "flow_id": email_flow_id,
            "code": "sample-email-code",
        },
    )
    assert complete_email.error is None
    assert isinstance(complete_email.result, dict)
    assert complete_email.result["ok"] is True

    invoke_email = await _rpc(
        server,
        request_id=5,
        method="capability.invoke",
        params={
            "context_token": user1_private,
            "capability": "gog.email",
            "operation": "list_messages",
            "input": {"folder": "inbox", "limit": 2},
        },
    )
    assert invoke_email.error is None
    assert isinstance(invoke_email.result, dict)
    email_output = invoke_email.result["output"]
    assert email_output["count"] == 2
    serialized_email_output = json.dumps(email_output, ensure_ascii=True).lower()
    assert "access_token" not in serialized_email_output
    assert "refresh_token" not in serialized_email_output
    assert "client_secret" not in serialized_email_output

    begin_calendar = await _rpc(
        server,
        request_id=6,
        method="capability.auth.begin",
        params={
            "context_token": user1_private,
            "capability": "gog.calendar",
            "account_hint": "work",
        },
    )
    assert begin_calendar.error is None
    assert isinstance(begin_calendar.result, dict)
    calendar_flow_id = str(begin_calendar.result["flow_id"])

    complete_calendar = await _rpc(
        server,
        request_id=7,
        method="capability.auth.complete",
        params={
            "context_token": user1_private,
            "flow_id": calendar_flow_id,
            "code": "sample-calendar-code",
        },
    )
    assert complete_calendar.error is None
    assert isinstance(complete_calendar.result, dict)
    assert complete_calendar.result["ok"] is True

    invoke_calendar = await _rpc(
        server,
        request_id=8,
        method="capability.invoke",
        params={
            "context_token": user1_private,
            "capability": "gog.calendar",
            "operation": "list_events",
            "input": {"window": "7d"},
        },
    )
    assert invoke_calendar.error is None
    assert isinstance(invoke_calendar.result, dict)
    calendar_output = invoke_calendar.result["output"]
    assert calendar_output["count"] == 1

    group_blocked = await _rpc(
        server,
        request_id=9,
        method="capability.invoke",
        params={
            "context_token": user1_group,
            "capability": "gog.email",
            "operation": "list_messages",
            "input": {"limit": 1},
        },
    )
    assert group_blocked.error is not None
    assert "capability_access_denied" in group_blocked.error.message

    user2_blocked = await _rpc(
        server,
        request_id=10,
        method="capability.invoke",
        params={
            "context_token": user2_private,
            "capability": "gog.email",
            "operation": "list_messages",
            "input": {"limit": 1},
        },
    )
    assert user2_blocked.error is not None
    assert "capability_auth_required" in user2_blocked.error.message

    state_text = state_path.read_text(encoding="utf-8")
    assert "sample-email-code" not in state_text
    assert "sample-calendar-code" not in state_text

    state_payload = json.loads(state_text)
    email_account = state_payload["accounts"]["user-1:gog.email:work"]
    calendar_account = state_payload["accounts"]["user-1:gog.calendar:work"]
    vault = FileVault(vault_path)

    email_vault_payload = vault.get_json(email_account["vault_ref"])
    calendar_vault_payload = vault.get_json(calendar_account["vault_ref"])
    assert isinstance(email_vault_payload, dict)
    assert isinstance(calendar_vault_payload, dict)
    assert email_vault_payload["auth_exchange"]["code"] == "sample-email-code"
    assert calendar_vault_payload["auth_exchange"]["code"] == "sample-calendar-code"
