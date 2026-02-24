#!/usr/bin/env python3
"""Reference gog capability bridge.

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
from urllib.parse import quote_plus

BRIDGE_VERSION = 1
BRIDGE_NAMESPACE = "gog"
TOKEN_TYPE = "ASH_CONTEXT"  # noqa: S105
TOKEN_ALG = "HS256"  # noqa: S105
TOKEN_LEEWAY_SECONDS = 30
ENV_CONTEXT_SECRET = "ASH_CONTEXT_TOKEN_SECRET"  # noqa: S105
ENV_STATE_PATH = "GOGCLI_STATE_PATH"
DEFAULT_STATE_PATH = Path.home() / ".ash" / "gogcli" / "state.json"


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


def _read_state() -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {"accounts": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"accounts": {}}
    if not isinstance(raw, dict):
        return {"accounts": {}}
    accounts = raw.get("accounts")
    if not isinstance(accounts, dict):
        return {"accounts": {}}
    return {"accounts": accounts}


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
    nonce = secrets.token_hex(8)
    flow_state = {
        "nonce": nonce,
        "user_id": claims.user_id,
        "capability_id": capability_id,
        "account_hint": account_hint,
        "issued_at": int(time.time()),
    }
    auth_url = (
        "https://auth.gog.local/authorize"
        f"?capability={quote_plus(capability_id)}"
        f"&account={quote_plus(account_hint)}"
        f"&nonce={quote_plus(nonce)}"
    )
    expires_at = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(int(time.time()) + 600),
    )
    return {
        "auth_url": auth_url,
        "expires_at": expires_at,
        "flow_state": flow_state,
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

    flow_user_id = _optional_text(flow_state.get("user_id"))
    if flow_user_id != claims.user_id:
        raise BridgeError(
            "capability_auth_flow_invalid",
            "flow_state user does not match caller",
        )

    account_ref = (
        _optional_text(flow_state.get("account_hint"))
        or _optional_text(params.get("account_hint"))
        or "default"
    )
    state = _read_state()
    account_key = _account_key(claims.user_id, capability_id, account_ref)
    state["accounts"][account_key] = {
        "created_at": int(time.time()),
        "provider": claims.provider,
        "chat_type": claims.chat_type,
        "credential_key": f"cred_{secrets.token_hex(8)}",
    }
    _write_state(state)
    return {
        "account_ref": account_ref,
        "credential_material": {
            "credential_key": state["accounts"][account_key]["credential_key"],
        },
        "metadata": {
            "provider": "google",
            "capability_id": capability_id,
        },
    }


def _require_linked_account(
    *,
    user_id: str,
    capability_id: str,
    account_ref: str,
) -> dict[str, Any]:
    state = _read_state()
    key = _account_key(user_id, capability_id, account_ref)
    account = state["accounts"].get(key)
    if not isinstance(account, dict):
        raise BridgeError(
            "capability_auth_required",
            "account is not linked for caller scope",
        )
    return account


def _as_object(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise BridgeError("capability_invalid_input", f"{field_name} must be an object")


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

    _ = _require_linked_account(
        user_id=claims.user_id,
        capability_id=capability_id,
        account_ref=account_ref,
    )
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if capability_id == "gog.email" and operation == "list_messages":
        folder = _optional_text(input_data.get("folder")) or "inbox"
        limit_value = input_data.get("limit", 10)
        try:
            limit = int(limit_value)
        except (TypeError, ValueError):
            raise BridgeError(
                "capability_invalid_input", "limit must be an integer"
            ) from None
        limit = max(1, min(limit, 50))
        messages = [
            {
                "id": f"msg_{index + 1}",
                "from": "updates@example.com",
                "subject": f"Sample message {index + 1}",
                "received_at": now_iso,
            }
            for index in range(limit)
        ]
        return {
            "output": {
                "folder": folder,
                "messages": messages,
                "count": len(messages),
                "account_ref": account_ref,
            }
        }

    if capability_id == "gog.email" and operation == "send_message":
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
        _required_text(
            input_data.get("body"),
            code="capability_invalid_input",
            message="body is required",
        )
        return {
            "output": {
                "status": "queued",
                "message_id": f"queued_{secrets.token_hex(6)}",
                "to": recipient,
                "subject": subject,
                "account_ref": account_ref,
            }
        }

    if capability_id == "gog.calendar" and operation == "list_events":
        window = _optional_text(input_data.get("window")) or "7d"
        events = [
            {
                "id": "evt_1",
                "title": "Example event",
                "start": now_iso,
                "calendar": _optional_text(input_data.get("calendar")) or "primary",
            }
        ]
        return {
            "output": {
                "window": window,
                "events": events,
                "count": len(events),
                "account_ref": account_ref,
            }
        }

    if capability_id == "gog.calendar" and operation == "create_event":
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
        return {
            "output": {
                "status": "created",
                "event_id": f"evt_{secrets.token_hex(6)}",
                "title": title,
                "start": start,
                "account_ref": account_ref,
            }
        }

    raise BridgeError(
        "capability_invalid_input",
        f"unsupported operation for {capability_id}: {operation}",
    )


def _dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "definitions":
        return _handle_definitions()
    if method == "auth_begin":
        return _handle_auth_begin(params)
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
