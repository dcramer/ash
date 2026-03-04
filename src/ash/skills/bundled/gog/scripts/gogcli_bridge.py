"""Bundled gog capability bridge.

Implements the bridge-v1 subprocess contract so Ash can call a namespaced
capability provider outside core runtime wiring.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from ash.config.paths import get_vault_path
from ash.security.vault import FileVault, VaultError

BRIDGE_VERSION = 1
BRIDGE_NAMESPACE = "gog"
TOKEN_TYPE = "ASH_CONTEXT"  # noqa: S105
TOKEN_ALG = "HS256"  # noqa: S105
TOKEN_LEEWAY_SECONDS = 30
ENV_CONTEXT_SECRET = "ASH_CONTEXT_TOKEN_SECRET"  # noqa: S105
ENV_STATE_PATH = "GOGCLI_STATE_PATH"
ENV_AUTH_FLOW_TTL_SECONDS = "GOGCLI_AUTH_FLOW_TTL_SECONDS"
ENV_VAULT_PATH = "GOGCLI_VAULT_PATH"
DEFAULT_STATE_PATH = Path.home() / ".ash" / "gogcli" / "state.json"
VAULT_NAMESPACE = "gog.credentials"
STATE_VERSION = 1
DEFAULT_AUTH_FLOW_TTL_SECONDS = 600
MIN_AUTH_FLOW_TTL_SECONDS = 30
MAX_AUTH_FLOW_TTL_SECONDS = 3600
ENV_GOOGLE_CLIENT_ID = "GOOGLE_CLIENT_ID"
ENV_GOOGLE_CLIENT_SECRET = "GOOGLE_CLIENT_SECRET"  # noqa: S105

ENV_GOOGLE_OAUTH_BASE_URL = "GOOGLE_OAUTH_BASE_URL"
DEFAULT_GOOGLE_OAUTH_BASE_URL = "https://oauth2.googleapis.com"
DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"

# Google authorization endpoint (different host from token endpoint)
DEFAULT_GOOGLE_AUTH_BASE_URL = "https://accounts.google.com"
ENV_GOOGLE_AUTH_BASE_URL = "GOOGLE_AUTH_BASE_URL"

# Scopes supported by device code flow (from Google docs)
DEVICE_CODE_ALLOWED_SCOPES: frozenset[str] = frozenset(
    {
        "email",
        "openid",
        "profile",
        "https://www.googleapis.com/auth/drive.appdata",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/youtube",
        "https://www.googleapis.com/auth/youtube.readonly",
    }
)

AUTH_CODE_REDIRECT_URI = "http://localhost"

ENV_GOOGLE_GMAIL_API_BASE_URL = "GOOGLE_GMAIL_API_BASE_URL"
DEFAULT_GOOGLE_GMAIL_API_BASE_URL = "https://gmail.googleapis.com"
ENV_GOOGLE_CALENDAR_API_BASE_URL = "GOOGLE_CALENDAR_API_BASE_URL"
DEFAULT_GOOGLE_CALENDAR_API_BASE_URL = "https://www.googleapis.com"

GMAIL_FOLDER_LABELS: dict[str, str] = {
    "inbox": "INBOX",
    "sent": "SENT",
    "drafts": "DRAFT",
    "spam": "SPAM",
    "trash": "TRASH",
    "starred": "STARRED",
    "important": "IMPORTANT",
    "unread": "UNREAD",
}

CAPABILITY_SCOPES: dict[str, str] = {
    "gog.email": (
        "https://www.googleapis.com/auth/gmail.readonly"
        " https://www.googleapis.com/auth/gmail.send"
    ),
    "gog.calendar": "https://www.googleapis.com/auth/calendar",
}


@dataclass(frozen=True, slots=True)
class VerifiedContext:
    """Verified caller claims extracted from context token."""

    user_id: str
    chat_id: str | None
    chat_type: str | None
    provider: str | None
    token_id: str | None


class BridgeError(ValueError):
    """Structured bridge error with stable capability error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    padded = text + "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _normalize_secret(value: str) -> bytes:
    text = value.strip()
    if not text:
        raise BridgeError(
            "capability_backend_unavailable",
            "ASH_CONTEXT_TOKEN_SECRET is empty",
        )

    if len(text) % 2 == 0:
        try:
            return bytes.fromhex(text)
        except ValueError:
            pass

    try:
        decoded = _b64url_decode(text)
        if decoded:
            return decoded
    except (binascii.Error, ValueError):
        pass

    return text.encode("utf-8")


def _required_text(value: Any, *, code: str, message: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise BridgeError(code, message)
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_claim(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return None


def _verify_context_token(token: str) -> VerifiedContext:
    secret_text = os.environ.get(ENV_CONTEXT_SECRET, "")
    if not secret_text:
        raise BridgeError(
            "capability_backend_unavailable",
            "ASH_CONTEXT_TOKEN_SECRET is not configured",
        )
    secret = _normalize_secret(secret_text)

    text = token.strip()
    parts = text.split(".")
    if len(parts) != 3:
        raise BridgeError("capability_invalid_input", "context_token format is invalid")

    encoded_header, encoded_payload, encoded_signature = parts
    signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")

    try:
        header = json.loads(_b64url_decode(encoded_header))
        payload = json.loads(_b64url_decode(encoded_payload))
        signature = _b64url_decode(encoded_signature)
    except (ValueError, TypeError, json.JSONDecodeError, binascii.Error):
        raise BridgeError(
            "capability_invalid_input", "context_token decode failed"
        ) from None

    if not isinstance(header, dict) or not isinstance(payload, dict):
        raise BridgeError(
            "capability_invalid_input", "context_token payload is invalid"
        )
    if header.get("alg") != TOKEN_ALG or header.get("typ") != TOKEN_TYPE:
        raise BridgeError("capability_invalid_input", "context_token header is invalid")

    expected = hmac.new(secret, signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise BridgeError(
            "capability_invalid_input", "context_token signature mismatch"
        )

    now = int(time.time())
    issued_at = _int_claim(payload, "iat")
    expires_at = _int_claim(payload, "exp")
    if issued_at is None or expires_at is None:
        raise BridgeError(
            "capability_invalid_input", "context_token time claims missing"
        )
    if issued_at - TOKEN_LEEWAY_SECONDS > now:
        raise BridgeError("capability_invalid_input", "context_token is not yet valid")
    if expires_at + TOKEN_LEEWAY_SECONDS < now:
        raise BridgeError("capability_invalid_input", "context_token expired")

    subject = _optional_text(payload.get("sub"))
    if not subject:
        raise BridgeError("capability_invalid_input", "context_token subject missing")

    return VerifiedContext(
        user_id=subject,
        chat_id=_optional_text(payload.get("chat_id")),
        chat_type=_optional_text(payload.get("chat_type")),
        provider=_optional_text(payload.get("provider")),
        token_id=_optional_text(payload.get("jti")),
    )


def _state_path() -> Path:
    configured = os.environ.get(ENV_STATE_PATH)
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_STATE_PATH


def _vault_path() -> Path:
    configured = _optional_text(os.environ.get(ENV_VAULT_PATH))
    if configured is not None:
        return Path(configured).expanduser()
    return get_vault_path()


def _vault() -> FileVault:
    return FileVault(_vault_path())


def _empty_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "accounts": {},
        "auth_flows": {},
        "operation_state": {},
    }


def _dict_values(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, dict):
            result[key] = dict(item)
    return result


def _read_state() -> dict[str, Any]:
    empty = _empty_state()
    path = _state_path()
    if not path.exists():
        return empty
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(raw, dict):
        return empty

    state = _empty_state()
    version = raw.get("version")
    if isinstance(version, int):
        state["version"] = version
    state["accounts"] = _dict_values(raw.get("accounts"))
    state["auth_flows"] = _dict_values(raw.get("auth_flows"))
    state["operation_state"] = _dict_values(raw.get("operation_state"))
    return state


def _write_state(state: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=str(path.parent),
        encoding="utf-8",
    ) as handle:
        json.dump(state, handle, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _account_key(user_id: str, capability_id: str, account_ref: str) -> str:
    return f"{user_id}:{capability_id}:{account_ref}"


def _operation_scope_key(user_id: str, capability_id: str) -> str:
    return f"{user_id}:{capability_id}"


def _auth_flow_ttl_seconds() -> int:
    value = _optional_text(os.environ.get(ENV_AUTH_FLOW_TTL_SECONDS))
    if value is None:
        return DEFAULT_AUTH_FLOW_TTL_SECONDS
    try:
        parsed = int(value)
    except ValueError:
        return DEFAULT_AUTH_FLOW_TTL_SECONDS
    return max(MIN_AUTH_FLOW_TTL_SECONDS, min(parsed, MAX_AUTH_FLOW_TTL_SECONDS))


def _prune_expired_flows(*, state: dict[str, Any], now_epoch: int) -> bool:
    auth_flows = _dict_values(state.get("auth_flows"))
    changed = False
    for flow_id, flow in list(auth_flows.items()):
        expires_at = _int_claim(flow, "expires_at")
        if expires_at is None or expires_at <= now_epoch:
            auth_flows.pop(flow_id, None)
            changed = True
    state["auth_flows"] = auth_flows
    return changed


def _iso8601_utc(epoch_seconds: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch_seconds))


def _require_namespaced_capability(capability_id: Any) -> str:
    capability = _required_text(
        capability_id,
        code="capability_invalid_input",
        message="capability_id is required",
    )
    if not capability.startswith(f"{BRIDGE_NAMESPACE}."):
        raise BridgeError(
            "capability_invalid_input",
            f"capability_id must be in {BRIDGE_NAMESPACE} namespace",
        )
    return capability


def _google_oauth_base_url() -> str:
    value = _optional_text(os.environ.get(ENV_GOOGLE_OAUTH_BASE_URL))
    return value or DEFAULT_GOOGLE_OAUTH_BASE_URL


def _google_device_code_url() -> str:
    return f"{_google_oauth_base_url()}/device/code"


def _google_token_url() -> str:
    return f"{_google_oauth_base_url()}/token"


def _scopes_support_device_code(scope_string: str) -> bool:
    return all(s in DEVICE_CODE_ALLOWED_SCOPES for s in scope_string.split())


def _google_authorization_url() -> str:
    value = _optional_text(os.environ.get(ENV_GOOGLE_AUTH_BASE_URL))
    base = value or DEFAULT_GOOGLE_AUTH_BASE_URL
    return f"{base}/o/oauth2/v2/auth"


def _google_client_id() -> str:
    value = _optional_text(os.environ.get(ENV_GOOGLE_CLIENT_ID))
    if not value:
        raise BridgeError(
            "capability_backend_unavailable",
            "GOOGLE_CLIENT_ID is not configured",
        )
    return value


def _google_client_secret() -> str:
    value = _optional_text(os.environ.get(ENV_GOOGLE_CLIENT_SECRET))
    if not value:
        raise BridgeError(
            "capability_backend_unavailable",
            "GOOGLE_CLIENT_SECRET is not configured",
        )
    return value


def _google_gmail_api_base_url() -> str:
    value = _optional_text(os.environ.get(ENV_GOOGLE_GMAIL_API_BASE_URL))
    return value or DEFAULT_GOOGLE_GMAIL_API_BASE_URL


def _google_calendar_api_base_url() -> str:
    value = _optional_text(os.environ.get(ENV_GOOGLE_CALENDAR_API_BASE_URL))
    return value or DEFAULT_GOOGLE_CALENDAR_API_BASE_URL


def _google_api_request(
    method: str,
    url: str,
    *,
    access_token: str,
    params: list[tuple[str, str]] | None = None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticated HTTP request to a Google API endpoint."""
    from urllib.error import HTTPError, URLError
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    if params:
        url = f"{url}?{urlencode(params, doseq=True)}"

    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")

    request = Request(url, data=data, method=method)  # noqa: S310
    request.add_header("Authorization", f"Bearer {access_token}")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urlopen(request, timeout=30) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        if e.code == 401:
            raise BridgeError(
                "capability_auth_required",
                "Google API returned 401 — re-authentication required",
            ) from None
        try:
            error_body = json.loads(e.read().decode("utf-8"))
        except (json.JSONDecodeError, OSError):
            error_body = {}
        error_msg = ""
        if isinstance(error_body.get("error"), dict):
            error_msg = _optional_text(error_body["error"].get("message")) or ""
        raise BridgeError(
            "capability_backend_unavailable",
            f"Google API returned HTTP {e.code}{': ' + error_msg if error_msg else ''}",
        ) from None
    except URLError as e:
        raise BridgeError(
            "capability_backend_unavailable",
            f"Google API request failed: {e.reason}",
        ) from None


def _get_access_token(vault_ref: str) -> str:
    """Read the current access_token from the vault."""
    try:
        vault = _vault()
        creds = vault.get_json(vault_ref)
    except VaultError:
        raise BridgeError(
            "capability_auth_required",
            "failed to read credentials from vault",
        ) from None
    if not isinstance(creds, dict):
        raise BridgeError(
            "capability_auth_required",
            "credentials not found in vault",
        )
    token = _optional_text(creds.get("access_token"))
    if not token:
        raise BridgeError(
            "capability_auth_required",
            "access_token not found in stored credentials",
        )
    return token


def _parse_window(window: str) -> int:
    """Parse a window string like '7d', '3h', '2w' into seconds."""
    if not window:
        return 7 * 86400  # default 7 days

    normalized = window.strip().lower()
    if len(normalized) < 2:
        raise BridgeError(
            "capability_invalid_input",
            "window must be a positive duration like '7d', '3h', or '2w'",
        )
    unit = normalized[-1]
    value_text = normalized[:-1]
    if not value_text.isdigit():
        raise BridgeError(
            "capability_invalid_input",
            "window must be a positive duration like '7d', '3h', or '2w'",
        )
    value = int(value_text)
    if value <= 0:
        raise BridgeError(
            "capability_invalid_input",
            "window must be greater than zero",
        )
    multipliers = {"h": 3600, "d": 86400, "w": 604800}
    multiplier = multipliers.get(unit)
    if multiplier is None:
        raise BridgeError(
            "capability_invalid_input",
            "window unit must be one of: h, d, w",
        )
    return value * multiplier


def _http_post_form(url: str, params: dict[str, str]) -> dict[str, Any]:
    """POST form-encoded data and return parsed JSON response."""
    from urllib.error import HTTPError, URLError
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    body = urlencode(params).encode("utf-8")
    request = Request(url, data=body, method="POST")  # noqa: S310
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urlopen(request, timeout=30) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        try:
            error_body = json.loads(e.read().decode("utf-8"))
        except (json.JSONDecodeError, OSError):
            raise BridgeError(
                "capability_backend_unavailable",
                f"Google API returned HTTP {e.code}",
            ) from None
        return error_body
    except URLError as e:
        raise BridgeError(
            "capability_backend_unavailable",
            f"Google API request failed: {e.reason}",
        ) from None


def _handle_definitions() -> dict[str, Any]:
    return {
        "definitions": [
            {
                "id": "gog.email",
                "description": "Google Mail operations",
                "sensitive": True,
                "allowed_chat_types": ["private"],
                "operations": [
                    {
                        "name": "list_messages",
                        "description": "List recent inbox messages",
                        "requires_auth": True,
                        "mutating": False,
                    },
                    {
                        "name": "send_message",
                        "description": "Send an email message",
                        "requires_auth": True,
                        "mutating": True,
                    },
                ],
            },
            {
                "id": "gog.calendar",
                "description": "Google Calendar operations",
                "sensitive": True,
                "allowed_chat_types": ["private"],
                "operations": [
                    {
                        "name": "list_events",
                        "description": "List calendar events",
                        "requires_auth": True,
                        "mutating": False,
                    },
                    {
                        "name": "create_event",
                        "description": "Create a calendar event",
                        "requires_auth": True,
                        "mutating": True,
                    },
                ],
            },
        ]
    }


def _handle_auth_begin(params: dict[str, Any]) -> dict[str, Any]:
    context_token = _required_text(
        params.get("context_token"),
        code="capability_invalid_input",
        message="context_token is required",
    )
    claims = _verify_context_token(context_token)

    capability_id = _require_namespaced_capability(params.get("capability_id"))
    account_hint = _optional_text(params.get("account_hint")) or "default"
    now_epoch = int(time.time())
    nonce = secrets.token_hex(8)
    flow_id = f"gaf_{secrets.token_hex(12)}"
    expires_epoch = now_epoch + _auth_flow_ttl_seconds()

    # Require Google OAuth credentials — fail loudly when not configured.
    client_id = _google_client_id()
    _google_client_secret()  # validate presence early
    scope = CAPABILITY_SCOPES.get(capability_id)
    if not scope:
        raise BridgeError(
            "capability_invalid_input",
            f"no OAuth scopes configured for {capability_id}",
        )

    if _scopes_support_device_code(scope):
        # Device code flow (RFC 8628)
        device_resp = _http_post_form(
            _google_device_code_url(),
            {"client_id": client_id, "scope": scope},
        )
        device_error = _optional_text(device_resp.get("error"))
        if device_error:
            error_desc = (
                _optional_text(device_resp.get("error_description")) or device_error
            )
            raise BridgeError(
                "capability_backend_unavailable",
                f"Google device code request failed: {error_desc}",
            )

        device_code = _optional_text(device_resp.get("device_code"))
        user_code = _optional_text(device_resp.get("user_code"))
        verification_url = _optional_text(device_resp.get("verification_url"))
        google_interval = _int_claim(device_resp, "interval") or 5
        google_expires_in = _int_claim(device_resp, "expires_in")

        if not device_code or not user_code or not verification_url:
            raise BridgeError(
                "capability_backend_unavailable",
                "Google device code response missing required fields",
            )

        if google_expires_in and google_expires_in < (expires_epoch - now_epoch):
            expires_epoch = now_epoch + google_expires_in

        state = _read_state()
        _prune_expired_flows(state=state, now_epoch=now_epoch)
        state["auth_flows"][flow_id] = {
            "user_id": claims.user_id,
            "capability_id": capability_id,
            "account_hint": account_hint,
            "nonce": nonce,
            "issued_at": now_epoch,
            "expires_at": expires_epoch,
            "device_code": device_code,
            "poll_interval": google_interval,
            "flow_type": "device_code",
        }
        _write_state(state)

        flow_state = {
            "flow_id": flow_id,
            "nonce": nonce,
            "device_code": device_code,
        }
        return {
            "auth_url": verification_url,
            "expires_at": _iso8601_utc(expires_epoch),
            "flow_state": flow_state,
            "flow_type": "device_code",
            "user_code": user_code,
            "poll_interval_seconds": google_interval,
        }

    # Authorization code flow (loopback redirect)
    from urllib.parse import urlencode

    # state_param is included in the auth URL for Google's CSRF protection.
    # The bridge cannot validate it on completion because the user pastes only
    # the auth code, not the full redirect URL.  Stored for audit/debugging.
    state_param = secrets.token_hex(16)
    auth_params = {
        "client_id": client_id,
        "redirect_uri": AUTH_CODE_REDIRECT_URI,
        "response_type": "code",
        "scope": scope,
        "access_type": "offline",
        "prompt": "consent",
        "state": state_param,
    }
    auth_url = f"{_google_authorization_url()}?{urlencode(auth_params)}"

    state = _read_state()
    _prune_expired_flows(state=state, now_epoch=now_epoch)
    state["auth_flows"][flow_id] = {
        "user_id": claims.user_id,
        "capability_id": capability_id,
        "account_hint": account_hint,
        "nonce": nonce,
        "issued_at": now_epoch,
        "expires_at": expires_epoch,
        "flow_type": "authorization_code",
        "state_param": state_param,
        "redirect_uri": AUTH_CODE_REDIRECT_URI,
    }
    _write_state(state)

    flow_state = {
        "flow_id": flow_id,
        "nonce": nonce,
    }
    return {
        "auth_url": auth_url,
        "expires_at": _iso8601_utc(expires_epoch),
        "flow_state": flow_state,
        "flow_type": "authorization_code",
    }


def _handle_auth_poll(params: dict[str, Any]) -> dict[str, Any]:
    context_token = _required_text(
        params.get("context_token"),
        code="capability_invalid_input",
        message="context_token is required",
    )
    claims = _verify_context_token(context_token)

    capability_id = _require_namespaced_capability(params.get("capability_id"))
    flow_state = params.get("flow_state")
    if not isinstance(flow_state, dict):
        raise BridgeError("capability_invalid_input", "flow_state must be an object")
    flow_id = _required_text(
        flow_state.get("flow_id"),
        code="capability_invalid_input",
        message="flow_state.flow_id is required",
    )
    device_code = _required_text(
        flow_state.get("device_code"),
        code="capability_invalid_input",
        message="flow_state.device_code is required",
    )

    state = _read_state()
    now_epoch = int(time.time())
    if _prune_expired_flows(state=state, now_epoch=now_epoch):
        _write_state(state)
    stored_flow = state["auth_flows"].get(flow_id)
    if not isinstance(stored_flow, dict):
        raise BridgeError(
            "capability_auth_flow_invalid",
            "flow is invalid or expired",
        )
    flow_user_id = _optional_text(stored_flow.get("user_id"))
    if flow_user_id != claims.user_id:
        raise BridgeError(
            "capability_auth_flow_invalid",
            "flow user does not match caller",
        )
    flow_capability_id = _optional_text(stored_flow.get("capability_id"))
    if flow_capability_id != capability_id:
        raise BridgeError(
            "capability_auth_flow_invalid",
            "flow capability does not match caller request",
        )
    expected_nonce = _optional_text(stored_flow.get("nonce"))
    nonce = _optional_text(flow_state.get("nonce"))
    if expected_nonce and nonce != expected_nonce:
        raise BridgeError(
            "capability_auth_flow_invalid",
            "flow_state nonce mismatch",
        )

    poll_interval = _int_claim(stored_flow, "poll_interval") or 5
    client_id = _google_client_id()
    client_secret = _google_client_secret()

    token_resp = _http_post_form(
        _google_token_url(),
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "device_code": device_code,
            "grant_type": DEVICE_CODE_GRANT_TYPE,
        },
    )

    # Check for pending/error responses
    error_code = _optional_text(token_resp.get("error"))
    if error_code == "authorization_pending":
        return {
            "status": "pending",
            "retry_after_seconds": poll_interval,
        }
    if error_code == "slow_down":
        return {
            "status": "pending",
            "retry_after_seconds": poll_interval + 5,
        }
    if error_code == "access_denied":
        state["auth_flows"].pop(flow_id, None)
        _write_state(state)
        raise BridgeError(
            "capability_auth_flow_denied",
            "user denied access during device code authorization",
        )
    if error_code == "expired_token":
        state["auth_flows"].pop(flow_id, None)
        _write_state(state)
        raise BridgeError(
            "capability_auth_flow_expired",
            "device code has expired — start a new auth flow",
        )
    if error_code:
        raise BridgeError(
            "capability_backend_unavailable",
            f"Google token endpoint returned error: {error_code}",
        )

    # Success — we got tokens
    access_token = _optional_text(token_resp.get("access_token"))
    refresh_token = _optional_text(token_resp.get("refresh_token"))
    if not access_token:
        raise BridgeError(
            "capability_backend_unavailable",
            "Google token response missing access_token",
        )

    account_hint = _optional_text(stored_flow.get("account_hint")) or "default"
    account_ref = account_hint
    account_key = _account_key(claims.user_id, capability_id, account_ref)

    existing = state["accounts"].get(account_key)
    existing_created_at = (
        _int_claim(existing, "created_at") if isinstance(existing, dict) else None
    )
    credential_key = (
        _optional_text(existing.get("credential_key"))
        if isinstance(existing, dict)
        else None
    )
    if credential_key is None:
        credential_key = f"cred_{secrets.token_hex(8)}"

    try:
        vault = _vault()
        vault_ref = vault.put_json(
            namespace=VAULT_NAMESPACE,
            key=account_key,
            payload={
                "credential_key": credential_key,
                "provider": "google",
                "capability_id": capability_id,
                "user_id": claims.user_id,
                "account_ref": account_ref,
                "linked_at": now_epoch,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": _optional_text(token_resp.get("token_type")),
                "expires_in": _int_claim(token_resp, "expires_in"),
                "obtained_at": now_epoch,
            },
        )
    except VaultError:
        raise BridgeError(
            "capability_backend_unavailable",
            "vault write failed during auth poll completion",
        ) from None

    state["accounts"][account_key] = {
        "created_at": existing_created_at or now_epoch,
        "updated_at": now_epoch,
        "provider": claims.provider,
        "chat_type": claims.chat_type,
        "credential_key": credential_key,
        "vault_ref": vault_ref,
    }
    state["auth_flows"].pop(flow_id, None)
    _write_state(state)

    return {
        "status": "complete",
        "account_ref": account_ref,
        "credential_material": {
            "credential_key": credential_key,
        },
        "metadata": {
            "provider": "google",
            "capability_id": capability_id,
            "linked_at": _iso8601_utc(now_epoch),
        },
    }


def _handle_auth_complete(params: dict[str, Any]) -> dict[str, Any]:
    context_token = _required_text(
        params.get("context_token"),
        code="capability_invalid_input",
        message="context_token is required",
    )
    claims = _verify_context_token(context_token)

    capability_id = _require_namespaced_capability(params.get("capability_id"))
    flow_state = params.get("flow_state")
    if not isinstance(flow_state, dict):
        raise BridgeError("capability_invalid_input", "flow_state must be an object")
    flow_id = _required_text(
        flow_state.get("flow_id"),
        code="capability_invalid_input",
        message="flow_state.flow_id is required",
    )

    state = _read_state()
    now_epoch = int(time.time())
    if _prune_expired_flows(state=state, now_epoch=now_epoch):
        _write_state(state)
    stored_flow = state["auth_flows"].get(flow_id)
    if not isinstance(stored_flow, dict):
        raise BridgeError(
            "capability_auth_flow_invalid",
            "flow_state is invalid or expired",
        )

    flow_user_id = _optional_text(stored_flow.get("user_id"))
    if flow_user_id != claims.user_id:
        raise BridgeError(
            "capability_auth_flow_invalid",
            "flow_state user does not match caller",
        )
    flow_capability_id = _optional_text(stored_flow.get("capability_id"))
    if flow_capability_id != capability_id:
        raise BridgeError(
            "capability_auth_flow_invalid",
            "flow_state capability does not match caller request",
        )
    expected_nonce = _optional_text(stored_flow.get("nonce"))
    nonce = _optional_text(flow_state.get("nonce"))
    if expected_nonce and nonce != expected_nonce:
        raise BridgeError(
            "capability_auth_flow_invalid",
            "flow_state nonce mismatch",
        )

    stored_flow_type = _optional_text(stored_flow.get("flow_type"))
    if stored_flow_type == "device_code":
        raise BridgeError(
            "capability_invalid_input",
            "device_code flows must use auth_poll, not auth_complete",
        )

    account_ref = (
        _optional_text(stored_flow.get("account_hint"))
        or _optional_text(params.get("account_hint"))
        or "default"
    )
    code = _required_text(
        params.get("code"),
        code="capability_invalid_input",
        message="code is required for auth_complete",
    )
    redirect_uri = (
        _optional_text(stored_flow.get("redirect_uri")) or AUTH_CODE_REDIRECT_URI
    )

    # Exchange authorization code for tokens
    client_id = _google_client_id()
    client_secret = _google_client_secret()
    token_resp = _http_post_form(
        _google_token_url(),
        {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        },
    )

    error_code = _optional_text(token_resp.get("error"))
    if error_code:
        error_desc = _optional_text(token_resp.get("error_description")) or error_code
        raise BridgeError(
            "capability_backend_unavailable",
            f"Google token exchange failed: {error_desc}",
        )

    access_token = _optional_text(token_resp.get("access_token"))
    refresh_token = _optional_text(token_resp.get("refresh_token"))
    if not access_token:
        raise BridgeError(
            "capability_backend_unavailable",
            "Google token response missing access_token",
        )

    account_key = _account_key(claims.user_id, capability_id, account_ref)
    existing = state["accounts"].get(account_key)
    existing_created_at = (
        _int_claim(existing, "created_at") if isinstance(existing, dict) else None
    )
    credential_key = (
        _optional_text(existing.get("credential_key"))
        if isinstance(existing, dict)
        else None
    )
    if credential_key is None:
        credential_key = f"cred_{secrets.token_hex(8)}"
    try:
        vault = _vault()
        vault_ref = vault.put_json(
            namespace=VAULT_NAMESPACE,
            key=account_key,
            payload={
                "credential_key": credential_key,
                "provider": "google",
                "capability_id": capability_id,
                "user_id": claims.user_id,
                "account_ref": account_ref,
                "linked_at": now_epoch,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": _optional_text(token_resp.get("token_type")),
                "expires_in": _int_claim(token_resp, "expires_in"),
                "obtained_at": now_epoch,
            },
        )
    except VaultError:
        raise BridgeError(
            "capability_backend_unavailable",
            "vault write failed during auth completion",
        ) from None
    state["accounts"][account_key] = {
        "created_at": existing_created_at or now_epoch,
        "updated_at": now_epoch,
        "provider": claims.provider,
        "chat_type": claims.chat_type,
        "credential_key": credential_key,
        "vault_ref": vault_ref,
    }
    state["auth_flows"].pop(flow_id, None)
    _write_state(state)
    return {
        "account_ref": account_ref,
        "credential_material": {
            "credential_key": credential_key,
        },
        "metadata": {
            "provider": "google",
            "capability_id": capability_id,
            "linked_at": _iso8601_utc(now_epoch),
        },
    }


def _require_linked_account(
    *,
    user_id: str,
    capability_id: str,
    account_ref: str,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state_data = state or _read_state()
    key = _account_key(user_id, capability_id, account_ref)
    account = state_data["accounts"].get(key)
    if not isinstance(account, dict):
        raise BridgeError(
            "capability_auth_required",
            "account is not linked for caller scope",
        )
    vault_ref = _optional_text(account.get("vault_ref"))
    if vault_ref:
        try:
            if _vault().get_json(vault_ref) is None:
                raise BridgeError(
                    "capability_auth_required",
                    "account credentials are unavailable for caller scope",
                )
        except VaultError:
            raise BridgeError(
                "capability_backend_unavailable",
                "vault read failed for linked account",
            ) from None
        return account

    # Backward compatibility for older pre-vault local state.
    if _optional_text(account.get("credential_key")):
        return account
    raise BridgeError(
        "capability_auth_required",
        "account is not linked for caller scope",
    )


def _as_object(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise BridgeError("capability_invalid_input", f"{field_name} must be an object")


def _refresh_token_if_needed(
    *,
    account_key: str,
    vault_ref: str | None,
    force: bool = False,
) -> bool:
    """Attempt to refresh Google access token if expired or explicitly forced."""
    if not vault_ref:
        return False
    try:
        vault = _vault()
        creds = vault.get_json(vault_ref)
    except VaultError:
        return False
    if not isinstance(creds, dict):
        return False
    refresh_token = _optional_text(creds.get("refresh_token"))
    if not refresh_token:
        return False
    obtained_at = _int_claim(creds, "obtained_at") or 0
    expires_in = _int_claim(creds, "expires_in") or 3600
    now_epoch = int(time.time())
    # Refresh 60 seconds before actual expiry
    if not force and now_epoch < obtained_at + expires_in - 60:
        return False
    try:
        client_id = _google_client_id()
        client_secret = _google_client_secret()
    except BridgeError:
        return False
    token_resp = _http_post_form(
        _google_token_url(),
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
    )
    new_access_token = _optional_text(token_resp.get("access_token"))
    if not new_access_token:
        return False
    creds["access_token"] = new_access_token
    new_expires_in = _int_claim(token_resp, "expires_in")
    if new_expires_in:
        creds["expires_in"] = new_expires_in
    creds["obtained_at"] = now_epoch
    try:
        vault.put_json(namespace=VAULT_NAMESPACE, key=account_key, payload=creds)
    except VaultError:
        return False
    return True


def _invoke_operation(
    *,
    capability_id: str,
    operation: str,
    input_data: dict[str, Any],
    access_token: str,
    account_ref: str,
) -> dict[str, Any]:
    if capability_id == "gog.email" and operation == "list_messages":
        return _invoke_list_messages(
            input_data=input_data,
            access_token=access_token,
            account_ref=account_ref,
        )

    if capability_id == "gog.email" and operation == "send_message":
        return _invoke_send_message(
            input_data=input_data,
            access_token=access_token,
            account_ref=account_ref,
        )

    if capability_id == "gog.calendar" and operation == "list_events":
        return _invoke_list_events(
            input_data=input_data,
            access_token=access_token,
            account_ref=account_ref,
        )

    if capability_id == "gog.calendar" and operation == "create_event":
        return _invoke_create_event(
            input_data=input_data,
            access_token=access_token,
            account_ref=account_ref,
        )

    raise BridgeError(
        "capability_invalid_input",
        f"unsupported operation for {capability_id}: {operation}",
    )


def _handle_invoke(params: dict[str, Any]) -> dict[str, Any]:
    context_token = _required_text(
        params.get("context_token"),
        code="capability_invalid_input",
        message="context_token is required",
    )
    claims = _verify_context_token(context_token)

    capability_id = _require_namespaced_capability(params.get("capability_id"))
    operation = _required_text(
        params.get("operation"),
        code="capability_invalid_input",
        message="operation is required",
    )
    account_ref = _required_text(
        params.get("account_ref"),
        code="capability_auth_required",
        message="account_ref is required",
    )
    input_data = _as_object(params.get("input_data"), field_name="input_data")

    state = _read_state()
    now_epoch = int(time.time())
    if _prune_expired_flows(state=state, now_epoch=now_epoch):
        _write_state(state)
    linked_account = _require_linked_account(
        user_id=claims.user_id,
        capability_id=capability_id,
        account_ref=account_ref,
        state=state,
    )
    linked_account_key = _account_key(claims.user_id, capability_id, account_ref)
    _refresh_token_if_needed(
        account_key=linked_account_key,
        vault_ref=_optional_text(linked_account.get("vault_ref")),
        force=False,
    )
    scope_key = _operation_scope_key(claims.user_id, capability_id)
    existing_scope = state["operation_state"].get(scope_key)
    invoke_count = 0
    if isinstance(existing_scope, dict):
        invoke_count = _int_claim(existing_scope, "invoke_count") or 0
    state["operation_state"][scope_key] = {
        "invoke_count": invoke_count + 1,
        "last_operation": operation,
        "last_account_ref": account_ref,
        "last_invoked_at": now_epoch,
    }
    _write_state(state)

    vault_ref = _optional_text(linked_account.get("vault_ref"))
    if not vault_ref:
        raise BridgeError(
            "capability_auth_required",
            "account has no vault reference",
        )
    access_token = _get_access_token(vault_ref)
    try:
        return _invoke_operation(
            capability_id=capability_id,
            operation=operation,
            input_data=input_data,
            access_token=access_token,
            account_ref=account_ref,
        )
    except BridgeError as e:
        # Access token can be stale due to clock drift/revocation despite local expiry checks.
        if e.code != "capability_auth_required":
            raise
        refreshed = _refresh_token_if_needed(
            account_key=linked_account_key,
            vault_ref=vault_ref,
            force=True,
        )
        if not refreshed:
            raise
        access_token = _get_access_token(vault_ref)
        return _invoke_operation(
            capability_id=capability_id,
            operation=operation,
            input_data=input_data,
            access_token=access_token,
            account_ref=account_ref,
        )


def _invoke_list_messages(
    *,
    input_data: dict[str, Any],
    access_token: str,
    account_ref: str,
) -> dict[str, Any]:
    folder = _optional_text(input_data.get("folder")) or "inbox"
    limit_value = input_data.get("limit", 10)
    try:
        limit = int(limit_value)
    except (TypeError, ValueError):
        raise BridgeError(
            "capability_invalid_input", "limit must be an integer"
        ) from None
    limit = max(1, min(limit, 50))

    label_id = GMAIL_FOLDER_LABELS.get(folder.lower(), folder.upper())
    gmail_base = _google_gmail_api_base_url()

    # Fetch message ID list
    list_resp = _google_api_request(
        "GET",
        f"{gmail_base}/gmail/v1/users/me/messages",
        access_token=access_token,
        params=[("labelIds", label_id), ("maxResults", str(limit))],
    )

    raw_messages = list_resp.get("messages")
    if not isinstance(raw_messages, list):
        raw_messages = []

    # Fetch metadata for each message
    messages: list[dict[str, Any]] = []
    for raw_msg in raw_messages[:limit]:
        msg_id = raw_msg.get("id") if isinstance(raw_msg, dict) else None
        if not msg_id:
            continue
        detail = _google_api_request(
            "GET",
            f"{gmail_base}/gmail/v1/users/me/messages/{msg_id}",
            access_token=access_token,
            params=[
                ("format", "metadata"),
                ("metadataHeaders", "From"),
                ("metadataHeaders", "Subject"),
                ("metadataHeaders", "Date"),
            ],
        )
        headers_list = (
            detail.get("payload", {}).get("headers", [])
            if isinstance(detail.get("payload"), dict)
            else []
        )
        header_map: dict[str, str] = {}
        for hdr in headers_list:
            if isinstance(hdr, dict):
                name = _optional_text(hdr.get("name"))
                value = _optional_text(hdr.get("value"))
                if name and value:
                    header_map[name.lower()] = value

        internal_date = _optional_text(detail.get("internalDate"))
        received_at = None
        if internal_date:
            try:
                received_at = _iso8601_utc(int(internal_date) // 1000)
            except (ValueError, OSError):
                received_at = header_map.get("date")
        else:
            received_at = header_map.get("date")

        messages.append(
            {
                "id": msg_id,
                "from": header_map.get("from", ""),
                "subject": header_map.get("subject", ""),
                "received_at": received_at or "",
                "snippet": _optional_text(detail.get("snippet")) or "",
            }
        )

    return {
        "output": {
            "folder": folder,
            "messages": messages,
            "count": len(messages),
            "account_ref": account_ref,
        }
    }


def _invoke_send_message(
    *,
    input_data: dict[str, Any],
    access_token: str,
    account_ref: str,
) -> dict[str, Any]:
    from email.mime.text import MIMEText

    recipient = _required_text(
        input_data.get("to"),
        code="capability_invalid_input",
        message="to is required",
    )
    subject = _required_text(
        input_data.get("subject"),
        code="capability_invalid_input",
        message="subject is required",
    )
    body = _required_text(
        input_data.get("body"),
        code="capability_invalid_input",
        message="body is required",
    )

    mime_msg = MIMEText(body)
    mime_msg["To"] = recipient
    mime_msg["Subject"] = subject
    raw_bytes = mime_msg.as_bytes()
    raw_b64 = base64.urlsafe_b64encode(raw_bytes).decode("ascii")

    gmail_base = _google_gmail_api_base_url()
    send_resp = _google_api_request(
        "POST",
        f"{gmail_base}/gmail/v1/users/me/messages/send",
        access_token=access_token,
        json_body={"raw": raw_b64},
    )

    return {
        "output": {
            "status": "sent",
            "message_id": _optional_text(send_resp.get("id")) or "",
            "to": recipient,
            "subject": subject,
            "account_ref": account_ref,
        }
    }


def _invoke_list_events(
    *,
    input_data: dict[str, Any],
    access_token: str,
    account_ref: str,
) -> dict[str, Any]:
    window = _optional_text(input_data.get("window")) or "7d"
    calendar_id = _optional_text(input_data.get("calendar")) or "primary"
    limit_value = input_data.get("limit", 25)
    try:
        limit = int(limit_value)
    except (TypeError, ValueError):
        limit = 25
    limit = max(1, min(limit, 250))

    now_epoch = int(time.time())
    window_seconds = _parse_window(window)
    time_min = _iso8601_utc(now_epoch)
    time_max = _iso8601_utc(now_epoch + window_seconds)

    cal_base = _google_calendar_api_base_url()
    from urllib.parse import quote

    list_resp = _google_api_request(
        "GET",
        f"{cal_base}/calendar/v3/calendars/{quote(calendar_id, safe='')}/events",
        access_token=access_token,
        params=[
            ("timeMin", time_min),
            ("timeMax", time_max),
            ("singleEvents", "true"),
            ("orderBy", "startTime"),
            ("maxResults", str(limit)),
        ],
    )

    raw_items = list_resp.get("items")
    if not isinstance(raw_items, list):
        raw_items = []

    events: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        start_obj = item.get("start", {})
        end_obj = item.get("end", {})
        if isinstance(start_obj, dict):
            start_val = start_obj.get("dateTime") or start_obj.get("date") or ""
        else:
            start_val = ""
        if isinstance(end_obj, dict):
            end_val = end_obj.get("dateTime") or end_obj.get("date") or ""
        else:
            end_val = ""
        events.append(
            {
                "id": _optional_text(item.get("id")) or "",
                "title": _optional_text(item.get("summary")) or "",
                "start": start_val,
                "end": end_val,
                "calendar": calendar_id,
                "location": _optional_text(item.get("location")) or "",
                "description": _optional_text(item.get("description")) or "",
            }
        )

    return {
        "output": {
            "window": window,
            "events": events,
            "count": len(events),
            "account_ref": account_ref,
        }
    }


def _invoke_create_event(
    *,
    input_data: dict[str, Any],
    access_token: str,
    account_ref: str,
) -> dict[str, Any]:
    title = _required_text(
        input_data.get("title"),
        code="capability_invalid_input",
        message="title is required",
    )
    start = _required_text(
        input_data.get("start"),
        code="capability_invalid_input",
        message="start is required",
    )
    end = _optional_text(input_data.get("end"))
    description = _optional_text(input_data.get("description"))
    location = _optional_text(input_data.get("location"))
    calendar_id = _optional_text(input_data.get("calendar")) or "primary"

    is_all_day = len(start) == 10  # "2026-03-02" format

    if is_all_day:
        from datetime import date, timedelta

        start_body: dict[str, str] = {"date": start}
        if end:
            end_body: dict[str, str] = {"date": end}
        else:
            try:
                end_body = {
                    "date": (date.fromisoformat(start) + timedelta(days=1)).isoformat()
                }
            except ValueError:
                end_body = {"date": start}
    else:
        start_body = {"dateTime": start}
        if end:
            end_body = {"dateTime": end}
        else:
            # Default: start + 1 hour
            try:
                # Parse ISO datetime and add 1 hour
                from datetime import datetime, timedelta

                dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                end_dt = dt + timedelta(hours=1)
                end_body = {"dateTime": end_dt.isoformat()}
            except (ValueError, OSError):
                end_body = {"dateTime": start}

    event_body: dict[str, Any] = {
        "summary": title,
        "start": start_body,
        "end": end_body,
    }
    if description:
        event_body["description"] = description
    if location:
        event_body["location"] = location

    cal_base = _google_calendar_api_base_url()
    from urllib.parse import quote

    created = _google_api_request(
        "POST",
        f"{cal_base}/calendar/v3/calendars/{quote(calendar_id, safe='')}/events",
        access_token=access_token,
        json_body=event_body,
    )

    created_start = created.get("start", {})
    created_end = created.get("end", {})
    if isinstance(created_start, dict):
        start_val = created_start.get("dateTime") or created_start.get("date") or start
    else:
        start_val = start
    if isinstance(created_end, dict):
        end_val = created_end.get("dateTime") or created_end.get("date") or ""
    else:
        end_val = ""

    return {
        "output": {
            "status": "created",
            "event_id": _optional_text(created.get("id")) or "",
            "title": _optional_text(created.get("summary")) or title,
            "start": start_val,
            "end": end_val,
            "calendar": calendar_id,
            "location": _optional_text(created.get("location")) or "",
            "description": _optional_text(created.get("description")) or "",
            "account_ref": account_ref,
        }
    }


def _dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "definitions":
        return _handle_definitions()
    if method == "auth_begin":
        return _handle_auth_begin(params)
    if method == "auth_poll":
        return _handle_auth_poll(params)
    if method == "auth_complete":
        return _handle_auth_complete(params)
    if method == "invoke":
        return _handle_invoke(params)
    raise BridgeError("capability_invalid_input", f"unsupported method: {method}")


def _emit_response(response: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(response, ensure_ascii=True))
    sys.stdout.flush()


def run_bridge() -> int:
    raw_stdin = sys.stdin.read()
    request_id = ""
    try:
        request = json.loads(raw_stdin)
        if not isinstance(request, dict):
            raise BridgeError("capability_invalid_input", "request must be an object")
        request_id = _required_text(
            request.get("id"),
            code="capability_invalid_input",
            message="request id is required",
        )
        version = request.get("version")
        if version != BRIDGE_VERSION:
            raise BridgeError("capability_invalid_input", "unsupported bridge version")
        namespace = _required_text(
            request.get("namespace"),
            code="capability_invalid_input",
            message="namespace is required",
        )
        if namespace != BRIDGE_NAMESPACE:
            raise BridgeError(
                "capability_invalid_input",
                f"unsupported namespace: {namespace}",
            )
        method = _required_text(
            request.get("method"),
            code="capability_invalid_input",
            message="method is required",
        )
        params = request.get("params") or {}
        if not isinstance(params, dict):
            raise BridgeError("capability_invalid_input", "params must be an object")
        result = _dispatch(method, params)
        _emit_response(
            {
                "version": BRIDGE_VERSION,
                "id": request_id,
                "result": result,
            }
        )
        return 0
    except BridgeError as error:
        _emit_response(
            {
                "version": BRIDGE_VERSION,
                "id": request_id,
                "error": {
                    "code": error.code,
                    "message": str(error),
                },
            }
        )
        return 0
    except Exception:
        _emit_response(
            {
                "version": BRIDGE_VERSION,
                "id": request_id,
                "error": {
                    "code": "capability_backend_unavailable",
                    "message": "bridge runtime failure",
                },
            }
        )
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="gog reference bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bridge", help="Run bridge-v1 command protocol")
    args = parser.parse_args()
    if args.command == "bridge":
        return run_bridge()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
