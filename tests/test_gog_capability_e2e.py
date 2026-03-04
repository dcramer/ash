from __future__ import annotations

import base64
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from ash.capabilities import CapabilityManager
from ash.capabilities.providers import SubprocessCapabilityProvider
from ash.context_token import ContextTokenService
from ash.rpc.methods.capability import register_capability_methods
from ash.rpc.server import RPCServer
from ash.security.vault import FileVault


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


# ---------------------------------------------------------------------------
# Fake Google OAuth + API server
# ---------------------------------------------------------------------------

_FAKE_USER_CODE = "ABCD-EFGH"
_FAKE_VERIFICATION_URL = "https://www.google.com/device"
_FAKE_ACCESS_TOKEN = "ya29.fake-access-token"  # noqa: S105
_FAKE_REFRESH_TOKEN = "1//fake-refresh-token"  # noqa: S105

# Shared mutable state for the fake server.
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
    """Handles OAuth, Gmail API, and Calendar API endpoints with canned responses."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # suppress noisy request logging

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
        device_code = f"fake-device-code-{_device_code_counter}"
        self._json_response(
            200,
            {
                "device_code": device_code,
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
            # First poll per device code returns pending, second returns tokens.
            count = _poll_counts.get(device_code, 0)
            _poll_counts[device_code] = count + 1
            if count == 0:
                self._json_response(
                    428,
                    {"error": "authorization_pending"},
                )
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
            if code:
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
    """Start a local HTTP server mimicking Google OAuth + API endpoints."""
    global _device_code_counter  # noqa: PLW0603
    _device_code_counter = 0
    _poll_counts.clear()
    server = HTTPServer(("127.0.0.1", 0), _FakeGoogleOAuthHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.mark.asyncio
async def test_gog_capability_rpc_stack_round_trip_and_policy_enforcement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_google_oauth: str,
) -> None:
    state_path = tmp_path / "gogcli-state.json"
    vault_path = tmp_path / "vault"
    monkeypatch.setenv("GOGCLI_STATE_PATH", str(state_path))
    monkeypatch.setenv("GOGCLI_VAULT_PATH", str(vault_path))

    service = ContextTokenService(secret=b"gog-capability-e2e-secret-32-bytes")
    manager = CapabilityManager(auth_flow_ttl_seconds=300)
    provider = SubprocessCapabilityProvider(
        namespace="gog",
        command=[
            sys.executable,
            "-m",
            "ash.skills.bundled.gog.scripts.gogcli_bridge",
            "bridge",
        ],
        context_token_service=service,
        env={
            "GOOGLE_CLIENT_ID": "fake-client-id",
            "GOOGLE_CLIENT_SECRET": "fake-client-secret",
            "GOOGLE_OAUTH_BASE_URL": fake_google_oauth,
            "GOOGLE_AUTH_BASE_URL": fake_google_oauth,
            "GOOGLE_GMAIL_API_BASE_URL": fake_google_oauth,
            "GOOGLE_CALENDAR_API_BASE_URL": fake_google_oauth,
        },
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

    # ---- List capabilities (private vs group) ----
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

    # ---- Auth begin (authorization code flow) for email ----
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
    assert begin_email.result["flow_type"] == "authorization_code"
    assert begin_email.result.get("user_code") is None
    assert "response_type=code" in begin_email.result["auth_url"]
    email_flow_id = str(begin_email.result["flow_id"])

    # ---- Auth complete (exchange code for tokens) ----
    complete_email = await _rpc(
        server,
        request_id=4,
        method="capability.auth.complete",
        params={
            "context_token": user1_private,
            "flow_id": email_flow_id,
            "code": "fake-auth-code-email",
        },
    )
    assert complete_email.error is None
    assert isinstance(complete_email.result, dict)
    assert complete_email.result["ok"] is True

    # ---- Invoke email list_messages ----
    invoke_email = await _rpc(
        server,
        request_id=6,
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
    assert email_output["messages"][0]["from"] == "sender@example.com"
    assert "Subject for msg_1" in email_output["messages"][0]["subject"]
    assert email_output["messages"][0]["snippet"]  # non-empty
    serialized_email_output = json.dumps(email_output, ensure_ascii=True).lower()
    assert "access_token" not in serialized_email_output
    assert "refresh_token" not in serialized_email_output
    assert "client_secret" not in serialized_email_output

    # ---- Invoke email get_message ----
    invoke_email_message = await _rpc(
        server,
        request_id=61,
        method="capability.invoke",
        params={
            "context_token": user1_private,
            "capability": "gog.email",
            "operation": "get_message",
            "input": {"id": "msg_1"},
        },
    )
    assert invoke_email_message.error is None
    assert isinstance(invoke_email_message.result, dict)
    message_output = invoke_email_message.result["output"]
    assert message_output["id"] == "msg_1"
    assert message_output["thread_id"] == "thread_msg_1"
    assert message_output["body_text"] == "Plain body for msg_1"

    # ---- Auth begin + complete for calendar ----
    begin_calendar = await _rpc(
        server,
        request_id=7,
        method="capability.auth.begin",
        params={
            "context_token": user1_private,
            "capability": "gog.calendar",
            "account_hint": "work",
        },
    )
    assert begin_calendar.error is None
    assert isinstance(begin_calendar.result, dict)
    assert begin_calendar.result["flow_type"] == "authorization_code"
    calendar_flow_id = str(begin_calendar.result["flow_id"])

    # Complete calendar auth code flow
    complete_calendar = await _rpc(
        server,
        request_id=8,
        method="capability.auth.complete",
        params={
            "context_token": user1_private,
            "flow_id": calendar_flow_id,
            "code": "fake-auth-code-calendar",
        },
    )
    assert complete_calendar.error is None
    assert isinstance(complete_calendar.result, dict)
    assert complete_calendar.result["ok"] is True

    # ---- Invoke calendar list_events ----
    invoke_calendar = await _rpc(
        server,
        request_id=10,
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
    assert calendar_output["events"][0]["title"] == "Fake calendar event"
    assert calendar_output["events"][0]["location"] == "Conference Room"

    # ---- Policy: group chat blocked ----
    group_blocked = await _rpc(
        server,
        request_id=11,
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

    # ---- Policy: user2 not authed ----
    user2_blocked = await _rpc(
        server,
        request_id=12,
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

    # ---- Verify state does not leak secrets ----
    state_text = state_path.read_text(encoding="utf-8")
    assert _FAKE_ACCESS_TOKEN not in state_text
    assert _FAKE_REFRESH_TOKEN not in state_text
    assert "fake-client-secret" not in state_text

    state_payload = json.loads(state_text)
    email_account = state_payload["accounts"]["user-1:gog.email:work"]
    calendar_account = state_payload["accounts"]["user-1:gog.calendar:work"]
    vault = FileVault(vault_path)

    email_vault_payload = vault.get_json(email_account["vault_ref"])
    calendar_vault_payload = vault.get_json(calendar_account["vault_ref"])
    assert isinstance(email_vault_payload, dict)
    assert isinstance(calendar_vault_payload, dict)
    assert email_vault_payload["access_token"] == _FAKE_ACCESS_TOKEN
    assert calendar_vault_payload["access_token"] == _FAKE_ACCESS_TOKEN
