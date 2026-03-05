from __future__ import annotations

import base64
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from ash.capabilities import CapabilityError
from ash.capabilities.providers import (
    CapabilityAuthCompleteInput,
    CapabilityCallContext,
    SubprocessCapabilityProvider,
)
from ash.context_token import ContextTokenService
from ash.security.vault import FileVault

_BRIDGE_MODULE = "ash.skills.bundled.gog.scripts.gogcli_bridge"

# ---------------------------------------------------------------------------
# Fake Google OAuth + API server
# ---------------------------------------------------------------------------

_FAKE_USER_CODE = "ABCD-EFGH"
_FAKE_VERIFICATION_URL = "https://www.google.com/device"
_FAKE_ACCESS_TOKEN = "ya29.fake-access-token"  # noqa: S105
_FAKE_REFRESH_TOKEN = "1//fake-refresh-token"  # noqa: S105

_device_code_counter = 0
_poll_counts: dict[str, int] = {}


def _b64url_text(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _fake_message_detail(msg_id: str, thread_id: str) -> dict[str, Any]:
    return {
        "id": msg_id,
        "threadId": thread_id,
        "snippet": f"Preview of {msg_id}",
        "internalDate": "1709337600000",
        "labelIds": ["INBOX"],
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "To", "value": "me@example.com"},
                {"name": "Subject", "value": f"Subject for {msg_id}"},
                {"name": "Date", "value": "Sat, 02 Mar 2024 00:00:00 +0000"},
            ],
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": _b64url_text(f"Plain body for {msg_id}")},
                },
                {
                    "mimeType": "text/html",
                    "body": {"data": _b64url_text(f"<p>HTML body for {msg_id}</p>")},
                },
            ],
        },
    }


class _FakeGoogleOAuthHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def _check_auth(self) -> bool:
        auth = self.headers.get("Authorization", "")
        if auth != f"Bearer {_FAKE_ACCESS_TOKEN}":
            self._json_response(401, {"error": {"message": "Invalid credentials"}})
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if not self._check_auth():
            return

        if path.startswith("/gmail/v1/users/me/threads/"):
            thread_id = path.split("/")[-1]
            self._json_response(
                200,
                {
                    "id": thread_id,
                    "messages": [
                        _fake_message_detail("msg_1", thread_id),
                        _fake_message_detail("msg_2", thread_id),
                    ],
                },
            )
            return

        # /gmail/v1/users/me/messages/{id} (must check before list endpoint)
        if path.startswith("/gmail/v1/users/me/messages/"):
            msg_id = path.split("/")[-1]
            self._json_response(200, _fake_message_detail(msg_id, f"thread_{msg_id}"))
            return

        # /gmail/v1/users/me/messages (list)
        if path == "/gmail/v1/users/me/messages":
            max_results = int((qs.get("maxResults") or ["10"])[0])
            msg_ids = [
                {"id": f"msg_{i + 1}", "threadId": f"thread_{i + 1}"}
                for i in range(max_results)
            ]
            self._json_response(
                200, {"messages": msg_ids, "resultSizeEstimate": len(msg_ids)}
            )
            return

        # /calendar/v3/calendars/{calendarId}/events
        if "/calendar/v3/calendars/" in path and path.endswith("/events"):
            self._json_response(
                200,
                {
                    "items": [
                        {
                            "id": "evt_1",
                            "summary": "Fake calendar event",
                            "start": {"dateTime": "2026-03-02T10:00:00Z"},
                            "end": {"dateTime": "2026-03-02T11:00:00Z"},
                            "location": "Conference Room",
                            "description": "A test event",
                        }
                    ]
                },
            )
            return

        self._json_response(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        # Gmail send and Calendar create require Bearer auth
        if self.path.startswith("/gmail/") or self.path.startswith("/calendar/"):
            if not self._check_auth():
                return

            if self.path == "/gmail/v1/users/me/messages/send":
                request_body = json.loads(body)
                self._json_response(
                    200,
                    {
                        "id": "sent_msg_001",
                        "threadId": "thread_sent_001",
                        "labelIds": ["SENT"],
                    },
                )
                return

            # /calendar/v3/calendars/{calendarId}/events
            if "/calendar/v3/calendars/" in self.path and self.path.endswith("/events"):
                request_body = json.loads(body)
                self._json_response(
                    200,
                    {
                        "id": "created_evt_001",
                        "summary": request_body.get("summary", ""),
                        "start": request_body.get("start", {}),
                        "end": request_body.get("end", {}),
                        "location": request_body.get("location", ""),
                        "description": request_body.get("description", ""),
                    },
                )
                return

            self._json_response(404, {"error": {"message": "not found"}})
            return

        # OAuth form-encoded endpoints
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
        elif grant_type == "authorization_code":
            code = (params.get("code") or [""])[0]
            if code == "fake-auth-code-no-refresh":
                self._json_response(
                    200,
                    {
                        "access_token": _FAKE_ACCESS_TOKEN,
                        "token_type": "Bearer",
                        "expires_in": 3600,
                    },
                )
            elif code:
                self._json_response(
                    200,
                    {
                        "access_token": _FAKE_ACCESS_TOKEN,
                        "refresh_token": _FAKE_REFRESH_TOKEN,
                        "token_type": "Bearer",
                        "expires_in": 3600,
                    },
                )
            else:
                self._json_response(400, {"error": "invalid_grant"})
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
    email_def = next(
        item
        for item in definitions
        if isinstance(item, dict) and item.get("id") == "gog.email"
    )
    operations = {
        operation.get("name")
        for operation in email_def.get("operations", [])
        if isinstance(operation, dict)
    }
    assert {
        "list_messages",
        "search_messages",
        "get_message",
        "get_thread",
        "send_message",
    }.issubset(operations)


def test_bridge_auth_code_flow_and_user_scoped_invoke(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    """gog.email uses authorization code flow (gmail scopes not device-code-compatible)."""
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
        "GOOGLE_AUTH_BASE_URL": fake_google_oauth,
        "GOOGLE_GMAIL_API_BASE_URL": fake_google_oauth,
        "GOOGLE_CALENDAR_API_BASE_URL": fake_google_oauth,
    }
    user1_token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
    )

    # auth_begin returns authorization code flow (gmail scopes not device-code-compatible)
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
    assert result["flow_type"] == "authorization_code"
    assert "user_code" not in result
    assert "poll_interval_seconds" not in result
    auth_url = result["auth_url"]
    assert "response_type=code" in auth_url
    assert "redirect_uri=http" in auth_url
    assert "access_type=offline" in auth_url
    flow_state = result["flow_state"]
    assert flow_state["flow_id"]
    assert flow_state["nonce"]

    state_after_begin = _load_state(state_path)
    assert flow_state["flow_id"] in state_after_begin["auth_flows"]
    stored_flow = state_after_begin["auth_flows"][flow_state["flow_id"]]
    assert stored_flow["flow_type"] == "authorization_code"
    assert stored_flow["state_param"]
    # Flows should remain valid long enough for real-world consent latency.
    assert int(stored_flow["expires_at"]) - int(time.time()) >= 25 * 60

    # auth_complete exchanges code for tokens
    complete = _run_bridge(
        {
            "version": 1,
            "id": "req_auth_complete",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "authorization_code": "fake-auth-code-from-redirect",
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
    account_key = "user-1:gog.email:work"
    vault_ref = state_after_complete["accounts"][account_key]["vault_ref"]
    vault_payload = FileVault(vault_path).get_json(vault_ref)
    assert isinstance(vault_payload, dict)
    credential_key = str(vault_payload["credential_key"])
    assert credential_key.startswith("cred_")
    assert vault_payload["access_token"] == _FAKE_ACCESS_TOKEN
    assert vault_payload["refresh_token"] == _FAKE_REFRESH_TOKEN

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
    # Verify real API response fields
    msg = messages[0]
    assert msg["id"] == "msg_1"
    assert msg["from"] == "sender@example.com"
    assert "Subject for msg_1" in msg["subject"]
    assert msg["snippet"]  # non-empty
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


def test_bridge_auth_complete_rejects_reused_flow(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    """After a flow completes via auth_complete, re-completing returns an error."""
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    env = {
        "ASH_CONTEXT_TOKEN_SECRET": service.export_verifier_secret(),
        "GOGCLI_STATE_PATH": str(tmp_path / "gogcli-state.json"),
        "GOGCLI_VAULT_PATH": str(tmp_path / "vault"),
        "GOOGLE_CLIENT_ID": "fake-client-id",
        "GOOGLE_CLIENT_SECRET": "fake-client-secret",
        "GOOGLE_OAUTH_BASE_URL": fake_google_oauth,
        "GOOGLE_AUTH_BASE_URL": fake_google_oauth,
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
    assert begin["result"]["flow_type"] == "authorization_code"
    flow_state = begin["result"]["flow_state"]

    # First auth_complete: succeeds
    first = _run_bridge(
        {
            "version": 1,
            "id": "req_reuse_complete_1",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "authorization_code": "fake-auth-code",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in first

    # Second auth_complete: flow is consumed — should fail
    second = _run_bridge(
        {
            "version": 1,
            "id": "req_reuse_complete_2",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "authorization_code": "fake-auth-code",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert second["error"]["code"] == "capability_auth_flow_invalid"


def test_bridge_auth_complete_rejects_expired_flow(
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
        "GOOGLE_AUTH_BASE_URL": fake_google_oauth,
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

    complete = _run_bridge(
        {
            "version": 1,
            "id": "req_expired_complete",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "authorization_code": "fake-auth-code",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert complete["error"]["code"] == "capability_auth_flow_invalid"


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
        "GOOGLE_AUTH_BASE_URL": fake_google_oauth,
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

    # Complete auth code flow
    complete = _run_bridge(
        {
            "version": 1,
            "id": "req_vault_complete",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "authorization_code": "fake-auth-code",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in complete
    account_ref = complete["result"]["account_ref"]

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


def test_bridge_auth_code_flow_for_calendar(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    """auth_begin for gog.calendar returns authorization_code flow with auth URL."""
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
        "GOOGLE_AUTH_BASE_URL": fake_google_oauth,
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
            "id": "req_cal_begin",
            "namespace": "gog",
            "method": "auth_begin",
            "params": {
                "capability_id": "gog.calendar",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in begin
    result = begin["result"]
    assert result["flow_type"] == "authorization_code"
    assert "user_code" not in result
    assert "poll_interval_seconds" not in result
    auth_url = result["auth_url"]
    assert "response_type=code" in auth_url
    assert "scope=" in auth_url
    assert "calendar" in auth_url

    # Complete with code — tokens stored in vault
    flow_state = result["flow_state"]
    complete = _run_bridge(
        {
            "version": 1,
            "id": "req_cal_complete",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.calendar",
                "flow_state": flow_state,
                "authorization_code": "fake-cal-auth-code",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in complete
    account_ref = complete["result"]["account_ref"]
    assert account_ref == "default"

    # Verify vault contains tokens (not raw auth_exchange)
    state = _load_state(state_path)
    account_key = "user-1:gog.calendar:default"
    vault_ref = state["accounts"][account_key]["vault_ref"]
    vault_payload = FileVault(vault_path).get_json(vault_ref)
    assert isinstance(vault_payload, dict)
    assert vault_payload["access_token"] == _FAKE_ACCESS_TOKEN
    assert vault_payload["refresh_token"] == _FAKE_REFRESH_TOKEN
    assert "auth_exchange" not in vault_payload


def test_bridge_auth_code_complete_invalid_code(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    """auth_complete requires normalized authorization_code input."""
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    env = {
        "ASH_CONTEXT_TOKEN_SECRET": service.export_verifier_secret(),
        "GOGCLI_STATE_PATH": str(tmp_path / "gogcli-state.json"),
        "GOGCLI_VAULT_PATH": str(tmp_path / "vault"),
        "GOOGLE_CLIENT_ID": "fake-client-id",
        "GOOGLE_CLIENT_SECRET": "fake-client-secret",
        "GOOGLE_OAUTH_BASE_URL": fake_google_oauth,
        "GOOGLE_AUTH_BASE_URL": fake_google_oauth,
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
            "id": "req_invalid_code_begin",
            "namespace": "gog",
            "method": "auth_begin",
            "params": {
                "capability_id": "gog.email",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in begin
    flow_state = begin["result"]["flow_state"]

    # auth_complete without authorization_code
    response = _run_bridge(
        {
            "version": 1,
            "id": "req_invalid_code_complete",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert response["error"]["code"] == "capability_invalid_input"
    assert "authorization_code is required" in response["error"]["message"]


def test_bridge_auth_complete_rejects_callback_url_without_normalized_code(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    """auth_complete does not parse callback_url in bridge layer."""
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
        "GOOGLE_AUTH_BASE_URL": fake_google_oauth,
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
            "id": "req_callback_begin",
            "namespace": "gog",
            "method": "auth_begin",
            "params": {
                "capability_id": "gog.calendar",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in begin
    flow_state = begin["result"]["flow_state"]
    complete = _run_bridge(
        {
            "version": 1,
            "id": "req_callback_complete",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.calendar",
                "flow_state": flow_state,
                "callback_url": "http://localhost/?code=fake-cal-auth-code",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert complete["error"]["code"] == "capability_invalid_input"
    assert "authorization_code is required" in complete["error"]["message"]


def test_bridge_auth_complete_preserves_existing_refresh_token_when_omitted(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    """Re-auth keeps prior refresh token when token response omits refresh_token."""
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    env = {
        "ASH_CONTEXT_TOKEN_SECRET": service.export_verifier_secret(),
        "GOGCLI_STATE_PATH": str(tmp_path / "gogcli-state.json"),
        "GOGCLI_VAULT_PATH": str(tmp_path / "vault"),
        "GOOGLE_CLIENT_ID": "fake-client-id",
        "GOOGLE_CLIENT_SECRET": "fake-client-secret",
        "GOOGLE_OAUTH_BASE_URL": fake_google_oauth,
        "GOOGLE_AUTH_BASE_URL": fake_google_oauth,
    }
    user_token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
    )

    # Initial link stores refresh token.
    begin_1 = _run_bridge(
        {
            "version": 1,
            "id": "req_preserve_refresh_begin_1",
            "namespace": "gog",
            "method": "auth_begin",
            "params": {
                "capability_id": "gog.email",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in begin_1
    flow_state_1 = begin_1["result"]["flow_state"]
    complete_1 = _run_bridge(
        {
            "version": 1,
            "id": "req_preserve_refresh_complete_1",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state_1,
                "authorization_code": "fake-auth-code-initial",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in complete_1

    # Re-auth with auth code that yields no refresh_token in fake OAuth server.
    begin_2 = _run_bridge(
        {
            "version": 1,
            "id": "req_preserve_refresh_begin_2",
            "namespace": "gog",
            "method": "auth_begin",
            "params": {
                "capability_id": "gog.email",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in begin_2
    flow_state_2 = begin_2["result"]["flow_state"]
    complete_2 = _run_bridge(
        {
            "version": 1,
            "id": "req_preserve_refresh_complete_2",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state_2,
                "authorization_code": "fake-auth-code-no-refresh",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in complete_2

    state = _load_state(Path(env["GOGCLI_STATE_PATH"]))
    account_key = "user-1:gog.email:default"
    vault_ref = str(state["accounts"][account_key]["vault_ref"])
    vault_payload = FileVault(Path(env["GOGCLI_VAULT_PATH"])).get_json(vault_ref)
    assert isinstance(vault_payload, dict)
    assert vault_payload["refresh_token"] == _FAKE_REFRESH_TOKEN


def test_bridge_auth_complete_reuses_refresh_token_from_related_capability(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    """Calendar auth can inherit refresh token from prior Gmail link for same account."""
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    env = {
        "ASH_CONTEXT_TOKEN_SECRET": service.export_verifier_secret(),
        "GOGCLI_STATE_PATH": str(tmp_path / "gogcli-state.json"),
        "GOGCLI_VAULT_PATH": str(tmp_path / "vault"),
        "GOOGLE_CLIENT_ID": "fake-client-id",
        "GOOGLE_CLIENT_SECRET": "fake-client-secret",
        "GOOGLE_OAUTH_BASE_URL": fake_google_oauth,
        "GOOGLE_AUTH_BASE_URL": fake_google_oauth,
    }
    user_token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
    )

    email_begin = _run_bridge(
        {
            "version": 1,
            "id": "req_reuse_related_begin_email",
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
    assert "error" not in email_begin
    email_flow_state = email_begin["result"]["flow_state"]
    email_complete = _run_bridge(
        {
            "version": 1,
            "id": "req_reuse_related_complete_email",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.email",
                "flow_state": email_flow_state,
                "authorization_code": "fake-auth-code-initial",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in email_complete

    calendar_begin = _run_bridge(
        {
            "version": 1,
            "id": "req_reuse_related_begin_calendar",
            "namespace": "gog",
            "method": "auth_begin",
            "params": {
                "capability_id": "gog.calendar",
                "account_hint": "work",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in calendar_begin
    calendar_flow_state = calendar_begin["result"]["flow_state"]
    calendar_complete = _run_bridge(
        {
            "version": 1,
            "id": "req_reuse_related_complete_calendar",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.calendar",
                "flow_state": calendar_flow_state,
                "authorization_code": "fake-auth-code-no-refresh",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in calendar_complete

    state = _load_state(Path(env["GOGCLI_STATE_PATH"]))
    calendar_key = "user-1:gog.calendar:work"
    calendar_vault_ref = str(state["accounts"][calendar_key]["vault_ref"])
    calendar_vault_payload = FileVault(Path(env["GOGCLI_VAULT_PATH"])).get_json(
        calendar_vault_ref
    )
    assert isinstance(calendar_vault_payload, dict)
    assert calendar_vault_payload["refresh_token"] == _FAKE_REFRESH_TOKEN


@pytest.mark.asyncio
async def test_subprocess_provider_auth_code_round_trip(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    """End-to-end through SubprocessCapabilityProvider for auth code flow."""
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
            "GOOGLE_AUTH_BASE_URL": fake_google_oauth,
            "GOOGLE_GMAIL_API_BASE_URL": fake_google_oauth,
            "GOOGLE_CALENDAR_API_BASE_URL": fake_google_oauth,
        },
    )
    definitions = await provider.definitions()
    ids = {definition.id for definition in definitions}
    assert ids == {"gog.email", "gog.calendar"}

    user1 = _context("user-1")
    begin = await provider.auth_begin(
        capability_id="gog.calendar",
        account_hint="work",
        context=user1,
    )
    assert begin.flow_type == "authorization_code"
    assert begin.user_code is None
    assert "response_type=code" in begin.auth_url

    # Complete with auth code
    complete = await provider.auth_complete(
        capability_id="gog.calendar",
        flow_state=begin.flow_state,
        completion=CapabilityAuthCompleteInput(
            authorization_code="fake-auth-code-from-redirect",
        ),
        context=user1,
    )
    assert complete.account_ref == "work"

    output = await provider.invoke(
        capability_id="gog.calendar",
        operation="list_events",
        input_data={},
        account_ref=complete.account_ref,
        idempotency_key=None,
        context=user1,
    )
    assert output["count"] == 1
    assert output["events"][0]["title"] == "Fake calendar event"

    user2 = _context("user-2")
    with pytest.raises(CapabilityError) as exc_info:
        await provider.invoke(
            capability_id="gog.calendar",
            operation="list_events",
            input_data={},
            account_ref=complete.account_ref,
            idempotency_key=None,
            context=user2,
        )
    assert exc_info.value.code == "capability_auth_required"


def _auth_email_account(
    *,
    tmp_path: Path,
    fake_google_oauth: str,
    env: dict[str, str],
    user_token: str,
) -> str:
    """Helper: run auth_begin + auth_complete for gog.email, return account_ref."""
    begin = _run_bridge(
        {
            "version": 1,
            "id": "req_helper_begin",
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
    complete = _run_bridge(
        {
            "version": 1,
            "id": "req_helper_complete",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.email",
                "flow_state": flow_state,
                "authorization_code": "fake-auth-code",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in complete
    return complete["result"]["account_ref"]


def _auth_calendar_account(
    *,
    env: dict[str, str],
    user_token: str,
) -> str:
    """Helper: run auth_begin + auth_complete for gog.calendar, return account_ref."""
    begin = _run_bridge(
        {
            "version": 1,
            "id": "req_cal_helper_begin",
            "namespace": "gog",
            "method": "auth_begin",
            "params": {
                "capability_id": "gog.calendar",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in begin
    flow_state = begin["result"]["flow_state"]
    complete = _run_bridge(
        {
            "version": 1,
            "id": "req_cal_helper_complete",
            "namespace": "gog",
            "method": "auth_complete",
            "params": {
                "capability_id": "gog.calendar",
                "flow_state": flow_state,
                "authorization_code": "fake-auth-code",
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in complete
    return complete["result"]["account_ref"]


def _make_env(
    *,
    service: ContextTokenService,
    tmp_path: Path,
    fake_google_oauth: str,
) -> dict[str, str]:
    return {
        "ASH_CONTEXT_TOKEN_SECRET": service.export_verifier_secret(),
        "GOGCLI_STATE_PATH": str(tmp_path / "gogcli-state.json"),
        "GOGCLI_VAULT_PATH": str(tmp_path / "vault"),
        "GOOGLE_CLIENT_ID": "fake-client-id",
        "GOOGLE_CLIENT_SECRET": "fake-client-secret",
        "GOOGLE_OAUTH_BASE_URL": fake_google_oauth,
        "GOOGLE_AUTH_BASE_URL": fake_google_oauth,
        "GOOGLE_GMAIL_API_BASE_URL": fake_google_oauth,
        "GOOGLE_CALENDAR_API_BASE_URL": fake_google_oauth,
    }


def test_bridge_invoke_send_message(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    """send_message builds MIME, base64-encodes, and POSTs to Gmail API."""
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    env = _make_env(
        service=service, tmp_path=tmp_path, fake_google_oauth=fake_google_oauth
    )
    user_token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
    )
    account_ref = _auth_email_account(
        tmp_path=tmp_path,
        fake_google_oauth=fake_google_oauth,
        env=env,
        user_token=user_token,
    )

    invoke = _run_bridge(
        {
            "version": 1,
            "id": "req_send",
            "namespace": "gog",
            "method": "invoke",
            "params": {
                "capability_id": "gog.email",
                "operation": "send_message",
                "input_data": {
                    "to": "recipient@example.com",
                    "subject": "Test email",
                    "body": "Hello from the bridge test!",
                },
                "account_ref": account_ref,
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in invoke
    output = invoke["result"]["output"]
    assert output["status"] == "sent"
    assert output["message_id"] == "sent_msg_001"
    assert output["to"] == "recipient@example.com"
    assert output["subject"] == "Test email"


def test_bridge_invoke_get_message(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    env = _make_env(
        service=service, tmp_path=tmp_path, fake_google_oauth=fake_google_oauth
    )
    user_token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
    )
    account_ref = _auth_email_account(
        tmp_path=tmp_path,
        fake_google_oauth=fake_google_oauth,
        env=env,
        user_token=user_token,
    )

    invoke = _run_bridge(
        {
            "version": 1,
            "id": "req_get_message",
            "namespace": "gog",
            "method": "invoke",
            "params": {
                "capability_id": "gog.email",
                "operation": "get_message",
                "input_data": {"id": "msg_1"},
                "account_ref": account_ref,
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in invoke
    output = invoke["result"]["output"]
    assert output["id"] == "msg_1"
    assert output["thread_id"] == "thread_msg_1"
    assert output["body_text"] == "Plain body for msg_1"
    assert "body_html" not in output


def test_bridge_invoke_get_thread(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    env = _make_env(
        service=service, tmp_path=tmp_path, fake_google_oauth=fake_google_oauth
    )
    user_token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
    )
    account_ref = _auth_email_account(
        tmp_path=tmp_path,
        fake_google_oauth=fake_google_oauth,
        env=env,
        user_token=user_token,
    )
    invoke = _run_bridge(
        {
            "version": 1,
            "id": "req_get_thread",
            "namespace": "gog",
            "method": "invoke",
            "params": {
                "capability_id": "gog.email",
                "operation": "get_thread",
                "input_data": {"thread_id": "thread_1"},
                "account_ref": account_ref,
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in invoke
    output = invoke["result"]["output"]
    assert output["thread_id"] == "thread_1"
    assert output["count"] == 2
    assert output["messages"][0]["body_text"] == "Plain body for msg_1"


def test_bridge_invoke_search_messages(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    env = _make_env(
        service=service, tmp_path=tmp_path, fake_google_oauth=fake_google_oauth
    )
    user_token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
    )
    account_ref = _auth_email_account(
        tmp_path=tmp_path,
        fake_google_oauth=fake_google_oauth,
        env=env,
        user_token=user_token,
    )
    invoke = _run_bridge(
        {
            "version": 1,
            "id": "req_search_messages",
            "namespace": "gog",
            "method": "invoke",
            "params": {
                "capability_id": "gog.email",
                "operation": "search_messages",
                "input_data": {"query": "from:sender@example.com", "limit": 2},
                "account_ref": account_ref,
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in invoke
    output = invoke["result"]["output"]
    assert output["query"] == "from:sender@example.com"
    assert output["count"] == 2


def test_bridge_invoke_calendar_create_event(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    """create_event POSTs to Calendar API and returns created event."""
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    env = _make_env(
        service=service, tmp_path=tmp_path, fake_google_oauth=fake_google_oauth
    )
    user_token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
    )
    account_ref = _auth_calendar_account(env=env, user_token=user_token)

    invoke = _run_bridge(
        {
            "version": 1,
            "id": "req_create_event",
            "namespace": "gog",
            "method": "invoke",
            "params": {
                "capability_id": "gog.calendar",
                "operation": "create_event",
                "input_data": {
                    "title": "Team standup",
                    "start": "2026-03-02T10:00:00Z",
                    "end": "2026-03-02T10:30:00Z",
                    "description": "Daily sync",
                    "location": "Zoom",
                },
                "account_ref": account_ref,
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in invoke
    output = invoke["result"]["output"]
    assert output["status"] == "created"
    assert output["event_id"] == "created_evt_001"
    assert output["title"] == "Team standup"
    assert output["location"] == "Zoom"
    assert output["description"] == "Daily sync"


def test_bridge_invoke_refreshes_access_token_after_401(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    env = _make_env(
        service=service, tmp_path=tmp_path, fake_google_oauth=fake_google_oauth
    )
    user_token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
    )
    account_ref = _auth_email_account(
        tmp_path=tmp_path,
        fake_google_oauth=fake_google_oauth,
        env=env,
        user_token=user_token,
    )

    state = _load_state(Path(env["GOGCLI_STATE_PATH"]))
    account_key = f"user-1:gog.email:{account_ref}"
    vault_ref = str(state["accounts"][account_key]["vault_ref"])
    vault = FileVault(Path(env["GOGCLI_VAULT_PATH"]))
    creds = vault.get_json(vault_ref)
    assert isinstance(creds, dict)
    creds["access_token"] = "expired-token"
    creds["obtained_at"] = 9_999_999_999
    creds["expires_in"] = 3600
    vault.put_json(namespace="gog.credentials", key=account_key, payload=creds)

    invoke = _run_bridge(
        {
            "version": 1,
            "id": "req_retry_after_401",
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
    assert "error" not in invoke
    assert invoke["result"]["output"]["count"] == 1

    refreshed_creds = vault.get_json(vault_ref)
    assert isinstance(refreshed_creds, dict)
    assert refreshed_creds["access_token"] == _FAKE_ACCESS_TOKEN


def test_bridge_invoke_list_events_rejects_invalid_window(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    env = _make_env(
        service=service, tmp_path=tmp_path, fake_google_oauth=fake_google_oauth
    )
    user_token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
    )
    account_ref = _auth_calendar_account(env=env, user_token=user_token)

    invoke = _run_bridge(
        {
            "version": 1,
            "id": "req_bad_window",
            "namespace": "gog",
            "method": "invoke",
            "params": {
                "capability_id": "gog.calendar",
                "operation": "list_events",
                "input_data": {"window": "-1d"},
                "account_ref": account_ref,
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert invoke["error"]["code"] == "capability_invalid_input"
    assert "window" in invoke["error"]["message"]


def test_bridge_invoke_calendar_create_event_defaults_all_day_end_to_next_day(
    tmp_path: Path,
    fake_google_oauth: str,
) -> None:
    service = ContextTokenService(secret=b"bridge-test-secret-32-bytes....")
    env = _make_env(
        service=service, tmp_path=tmp_path, fake_google_oauth=fake_google_oauth
    )
    user_token = service.issue(
        effective_user_id="user-1",
        chat_id="chat-1",
        chat_type="private",
        provider="telegram",
    )
    account_ref = _auth_calendar_account(env=env, user_token=user_token)

    invoke = _run_bridge(
        {
            "version": 1,
            "id": "req_create_all_day_event",
            "namespace": "gog",
            "method": "invoke",
            "params": {
                "capability_id": "gog.calendar",
                "operation": "create_event",
                "input_data": {
                    "title": "All day event",
                    "start": "2026-03-02",
                },
                "account_ref": account_ref,
                "context_token": user_token,
            },
        },
        env=env,
    )
    assert "error" not in invoke
    output = invoke["result"]["output"]
    assert output["start"] == "2026-03-02"
    assert output["end"] == "2026-03-03"
