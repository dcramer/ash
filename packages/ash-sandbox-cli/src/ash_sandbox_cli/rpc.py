"""RPC client for sandbox-to-host communication."""

import base64
import binascii
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from ash_rpc_protocol import (
    RPCRequest,
    RPCResponse,
    read_message_sync,
)

DEFAULT_SOCKET_PATH = "/opt/ash/rpc.sock"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 0.5  # seconds
CONTEXT_TOKEN_ENV = "ASH_CONTEXT_TOKEN"  # noqa: S105


class RPCError(Exception):
    """RPC call failed."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


def rpc_call(
    method: str,
    params: dict[str, Any] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> Any:
    """Make an RPC call to the host Ash process.

    Args:
        method: RPC method name (e.g., "memory.search").
        params: Method parameters.
        max_retries: Maximum number of retry attempts for connection errors.
        retry_delay: Delay between retries in seconds.

    Returns:
        The result from the RPC call.

    Raises:
        RPCError: If the RPC call fails.
        ConnectionError: If unable to connect to the RPC server after retries.
    """
    socket_path = os.environ.get("ASH_RPC_SOCKET", DEFAULT_SOCKET_PATH)

    if not Path(socket_path).exists():
        raise ConnectionError(f"RPC socket not found: {socket_path}")

    context_token = (os.environ.get(CONTEXT_TOKEN_ENV) or "").strip()
    if not context_token:
        raise ConnectionError(
            "Missing ASH_CONTEXT_TOKEN for sandbox RPC authentication"
        )

    request_params = dict(params or {})
    request_params["context_token"] = context_token

    # Create request
    request = RPCRequest(method=method, params=request_params)

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(socket_path)
            sock.sendall(request.to_bytes())

            # Read response
            data = read_message_sync(sock)
            if data is None:
                raise ConnectionError("Connection closed by server")

            # Parse response
            response = RPCResponse.from_dict(json.loads(data))

            if response.error:
                raise RPCError(
                    code=response.error.code,
                    message=response.error.message,
                    data=response.error.data,
                )

            return response.result

        except (ConnectionError, OSError, json.JSONDecodeError) as e:
            # Retry on connection errors and corrupt responses
            last_error = e
            if attempt < max_retries:
                time.sleep(retry_delay)
            # Continue to next attempt
        finally:
            sock.close()

    # All retries exhausted
    raise ConnectionError(
        f"RPC connection failed after {max_retries + 1} attempts: {last_error}"
    )


def _b64url_decode(text: str) -> bytes:
    padded = text + "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _decode_context_claims() -> dict[str, Any]:
    """Decode (without verifying) context token payload for local CLI context."""
    token = (os.environ.get(CONTEXT_TOKEN_ENV) or "").strip()
    if not token:
        return {}

    parts = token.split(".")
    if len(parts) != 3:
        return {}

    try:
        payload = json.loads(_b64url_decode(parts[1]))
    except (ValueError, TypeError, json.JSONDecodeError, binascii.Error):
        return {}

    return payload if isinstance(payload, dict) else {}


def _claim_str(claims: dict[str, Any], key: str) -> str | None:
    value = claims.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def get_context_params() -> dict[str, str | None]:
    """Get user/chat context from signed token claims.

    Returns:
        Dict with user_id, chat_id, and speaker attribution from context token.
    """
    claims = _decode_context_claims()
    source_username = _claim_str(claims, "source_username")

    return {
        "user_id": _claim_str(claims, "sub"),
        "chat_id": _claim_str(claims, "chat_id"),
        "chat_type": _claim_str(claims, "chat_type"),
        "chat_title": _claim_str(claims, "chat_title"),
        "thread_id": _claim_str(claims, "thread_id"),
        "source_username": source_username,
        "source_display_name": _claim_str(claims, "source_display_name"),
        "message_id": _claim_str(claims, "message_id"),
        "session_key": _claim_str(claims, "session_key"),
        "current_user_message": _claim_str(claims, "current_user_message"),
        "provider": _claim_str(claims, "provider"),
        "timezone": _claim_str(claims, "timezone"),
        # Convenience aliases for existing command code paths.
        "username": source_username,
    }
