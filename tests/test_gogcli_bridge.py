from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import pytest

from ash.capabilities import CapabilityError
from ash.capabilities.providers import (
    CapabilityCallContext,
    SubprocessCapabilityProvider,
)
from ash.context_token import ContextTokenService
from ash.security.vault import FileVault

_BRIDGE_MODULE = "ash.skills.bundled.gog.scripts.gogcli_bridge"

# ---------------------------------------------------------------------------
# Fake Google OAuth server (shared with e2e test)
# ---------------------------------------------------------------------------

_FAKE_USER_CODE = "ABCD-EFGH"
_FAKE_VERIFICATION_URL = "https://www.google.com/device"
_FAKE_ACCESS_TOKEN = "ya29.fake-access-token"  # noqa: S105
_FAKE_REFRESH_TOKEN = "1//fake-refresh-token"  # noqa: S105

_device_code_counter = 0
_poll_counts: dict[str, int] = {}


class _FakeGoogleOAuthHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        params = parse_qs(body)

        if self.path == "/device/code":
            self._handle_device_code(params)
        elif self.path == "/token":
            self._handle_token(params)
        else:
            self._json_response(404, {"error": "not_found"})

    def _handle_device_code(self, params: dict[str, list[str]]) -> None:
        global _device_code_counter  # noqa: PLW0603
        _device_code_counter += 1
        self._json_response(
            200,
            {
                "device_code": f"fake-device-code-{_device_code_counter}",
                "user_code": _FAKE_USER_CODE,
                "verification_url": _FAKE_VERIFICATION_URL,
                "expires_in": 1800,
                "interval": 1,
            },
        )

    def _handle_token(self, params: dict[str, list[str]]) -> None:
        grant_type = (params.get("grant_type") or [""])[0]
        device_code = (params.get("device_code") or [""])[0]

        if grant_type == "urn:ietf:params:oauth:grant-type:device_code":
            # First poll returns pending, second returns tokens.
            count = _poll_counts.get(device_code, 0)
            _poll_counts[device_code] = count + 1
            if count == 0:
                self._json_response(428, {"error": "authorization_pending"})
            else:
                self._json_response(
                    200,
                    {
                        "access_token": _FAKE_ACCESS_TOKEN,
                        "refresh_token": _FAKE_REFRESH_TOKEN,
                        "token_type": "Bearer",
                        "expires_in": 3600,
                    },
                )
        elif grant_type == "refresh_token":
            self._json_response(
                200,
                {
                    "access_token": _FAKE_ACCESS_TOKEN,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                },
            )
        else:
            self._json_response(400, {"error": "unsupported_grant_type"})

    def _json_response(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture()
def fake_google_oauth():
    global _device_code_counter  # noqa: PLW0603
    _device_code_counter = 0
    _poll_counts.clear()
    server = HTTPServer(("127.0.0.1", 0), _FakeGoogleOAuthHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


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
        [sys.executable, "-m", _BRIDGE_MODULE, "bridge"],  # noqa: S607
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


def test_bridge_auth_flow_and_user_scoped_invoke(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    state_path = tmp_path / "gogcli-state.json"
    vault_path = tmp_path / "vault"
    env = {
        "ASH_CONTEXT_TOKEN_SECRET": service.export_verifier_secret(),
        "GOGCLI_STATE_PATH": str(state_path),
        "GOGCLI_VAULT_PATH": str(vault_path),
        "GOOGLE_CLIENT_ID": "fake-client-id",
        "GOOGLE_CLIENT_SECRET": "fake-client-secret",
        "GOOGLE_OAUTH_BASE_URL": fake_google_oauth,
    }
    user1_token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
    )

    # auth_begin returns device code flow
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
    result = begin["result"]
    assert result["flow_type"] == "device_code"
    assert result["user_code"] == _FAKE_USER_CODE
    flow_state = result["flow_state"]
    assert flow_state["flow_id"]
    assert flow_state["nonce"]
    assert flow_state["device_code"]

    state_after_begin = _load_state(state_path)
    assert flow_state["flow_id"] in state_after_begin["auth_flows"]

    # First poll: pending
    poll1 = _run_bridge(
        {
            "version": 1,
            "id": "req_auth_poll_1",
            "namespace": "gog",
            "method": "auth_poll",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "context_token": user1_token,
            },
        },
        env=env,
    )
    assert "error" not in poll1
    assert poll1["result"]["status"] == "pending"

    # Second poll: complete with tokens
    poll2 = _run_bridge(
        {
            "version": 1,
            "id": "req_auth_poll_2",
            "namespace": "gog",
            "method": "auth_poll",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "context_token": user1_token,
            },
        },
        env=env,
    )
    assert "error" not in poll2
    assert poll2["result"]["status"] == "complete"
    account_ref = poll2["result"]["account_ref"]
    assert account_ref == "work"

    state_after_poll = _load_state(state_path)
    assert flow_state["flow_id"] not in state_after_poll["auth_flows"]
    account_key = "user-1:gog.email:work"
    vault_ref = state_after_poll["accounts"][account_key]["vault_ref"]
    vault_payload = FileVault(vault_path).get_json(vault_ref)
    assert isinstance(vault_payload, dict)
    credential_key = str(vault_payload["credential_key"])
    assert credential_key.startswith("cred_")
    assert vault_payload["access_token"] == _FAKE_ACCESS_TOKEN

    # Invoke with authed account
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
    assert account_key in state_after_invoke["accounts"]
    scope_key = "user-1:gog.email"
    assert state_after_invoke["operation_state"][scope_key]["invoke_count"] == 1

    # Different user is rejected
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


def test_bridge_auth_begin_fails_without_credentials(tmp_path: Path) -> None:
    """auth_begin fails loudly when GOOGLE_CLIENT_ID is missing."""
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    env = {
        "ASH_CONTEXT_TOKEN_SECRET": service.export_verifier_secret(),
        "GOGCLI_STATE_PATH": str(tmp_path / "gogcli-state.json"),
        "GOGCLI_VAULT_PATH": str(tmp_path / "vault"),
    }
    user_token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
    )

    response = _run_bridge(
        {
            "version": 1,
            "id": "req_no_creds",
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
    assert response["error"]["code"] == "capability_backend_unavailable"
    assert "GOOGLE_CLIENT_ID" in response["error"]["message"]


def test_bridge_auth_poll_rejects_reused_flow(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    """After a flow completes via auth_poll, re-polling returns an error."""
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    env = {
        "ASH_CONTEXT_TOKEN_SECRET": service.export_verifier_secret(),
        "GOGCLI_STATE_PATH": str(tmp_path / "gogcli-state.json"),
        "GOGCLI_VAULT_PATH": str(tmp_path / "vault"),
        "GOOGLE_CLIENT_ID": "fake-client-id",
        "GOOGLE_CLIENT_SECRET": "fake-client-secret",
        "GOOGLE_OAUTH_BASE_URL": fake_google_oauth,
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

    # First poll: pending
    _run_bridge(
        {
            "version": 1,
            "id": "req_reuse_poll_1",
            "namespace": "gog",
            "method": "auth_poll",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "context_token": user_token,
            },
        },
        env=env,
    )

    # Second poll: completes and consumes the flow
    second = _run_bridge(
        {
            "version": 1,
            "id": "req_reuse_poll_2",
            "namespace": "gog",
            "method": "auth_poll",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in second
    assert second["result"]["status"] == "complete"

    # Third poll: flow is consumed — should fail
    third = _run_bridge(
        {
            "version": 1,
            "id": "req_reuse_poll_3",
            "namespace": "gog",
            "method": "auth_poll",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert third["error"]["code"] == "capability_auth_flow_invalid"


def test_bridge_auth_poll_rejects_expired_flow(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    state_path = tmp_path / "gogcli-state.json"
    env = {
        "ASH_CONTEXT_TOKEN_SECRET": service.export_verifier_secret(),
        "GOGCLI_STATE_PATH": str(state_path),
        "GOGCLI_VAULT_PATH": str(tmp_path / "vault"),
        "GOOGLE_CLIENT_ID": "fake-client-id",
        "GOOGLE_CLIENT_SECRET": "fake-client-secret",
        "GOOGLE_OAUTH_BASE_URL": fake_google_oauth,
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

    # Force-expire the flow
    state = _load_state(state_path)
    state["auth_flows"][flow_id]["expires_at"] = 1
    state_path.write_text(json.dumps(state, ensure_ascii=True), encoding="utf-8")

    poll = _run_bridge(
        {
            "version": 1,
            "id": "req_expired_poll",
            "namespace": "gog",
            "method": "auth_poll",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert poll["error"]["code"] == "capability_auth_flow_invalid"


def test_bridge_invoke_requires_vault_record(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    state_path = tmp_path / "gogcli-state.json"
    vault_path = tmp_path / "vault"
    env = {
        "ASH_CONTEXT_TOKEN_SECRET": service.export_verifier_secret(),
        "GOGCLI_STATE_PATH": str(state_path),
        "GOGCLI_VAULT_PATH": str(vault_path),
        "GOOGLE_CLIENT_ID": "fake-client-id",
        "GOOGLE_CLIENT_SECRET": "fake-client-secret",
        "GOOGLE_OAUTH_BASE_URL": fake_google_oauth,
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
            "id": "req_vault_begin",
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

    # Pending then complete
    _run_bridge(
        {
            "version": 1,
            "id": "req_vault_poll_1",
            "namespace": "gog",
            "method": "auth_poll",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "context_token": user_token,
            },
        },
        env=env,
    )
    poll2 = _run_bridge(
        {
            "version": 1,
            "id": "req_vault_poll_2",
            "namespace": "gog",
            "method": "auth_poll",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in poll2
    account_ref = poll2["result"]["account_ref"]

    # Delete the vault entry
    state = _load_state(state_path)
    account = state["accounts"]["user-1:gog.email:work"]
    assert FileVault(vault_path).delete(account["vault_ref"]) is True

    invoke = _run_bridge(
        {
            "version": 1,
            "id": "req_vault_invoke",
            "namespace": "gog",
            "method": "invoke",
            "params": {
                "capability_id": "gog.email",
                "operation": "list_messages",
                "input_data": {"limit": 1},
                "account_ref": account_ref,
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert invoke["error"]["code"] == "capability_auth_required"


def test_bridge_rejects_invalid_context_signature(tmp_path: Path) -> None:
    signer = ContextTokenService(secret=b"bridge-signing-secret-32-bytes...")
    verifier = ContextTokenService(secret=b"bridge-verifier-secret-32-bytes..")
    env = {
        "ASH_CONTEXT_TOKEN_SECRET": verifier.export_verifier_secret(),
        "GOGCLI_STATE_PATH": str(tmp_path / "gogcli-state.json"),
        "GOGCLI_VAULT_PATH": str(tmp_path / "vault"),
        "GOOGLE_CLIENT_ID": "fake-client-id",
        "GOOGLE_CLIENT_SECRET": "fake-client-secret",
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
async def test_subprocess_provider_round_trip_with_bridge(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    service = ContextTokenService(secret=b"provider-roundtrip-secret-32-bytes")
    provider = SubprocessCapabilityProvider(
        namespace="gog",
        command=[sys.executable, "-m", _BRIDGE_MODULE, "bridge"],
        context_token_service=service,
        env={
            "GOGCLI_STATE_PATH": str(tmp_path / "gogcli-state.json"),
            "GOGCLI_VAULT_PATH": str(tmp_path / "vault"),
            "GOOGLE_CLIENT_ID": "fake-client-id",
            "GOOGLE_CLIENT_SECRET": "fake-client-secret",
            "GOOGLE_OAUTH_BASE_URL": fake_google_oauth,
        },
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
    assert begin.flow_type == "device_code"
    assert begin.user_code == _FAKE_USER_CODE

    # First poll: pending
    poll1 = await provider.auth_poll(
        capability_id="gog.email",
        flow_state=begin.flow_state,
        context=user1,
    )
    assert poll1.status == "pending"

    # Second poll: complete
    poll2 = await provider.auth_poll(
        capability_id="gog.email",
        flow_state=begin.flow_state,
        context=user1,
    )
    assert poll2.status == "complete"
    assert poll2.account_ref == "work"

    output = await provider.invoke(
        capability_id="gog.email",
        operation="list_messages",
        input_data={"limit": 1},
        account_ref=poll2.account_ref,
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
            account_ref=poll2.account_ref,
            idempotency_key=None,
            context=user2,
        )
    assert exc_info.value.code == "capability_auth_required"
