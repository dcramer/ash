"""RPC client for sandbox-to-host communication."""

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

    # Create request
    request = RPCRequest(method=method, params=params or {})

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


def get_context_params() -> dict[str, str | None]:
    """Get user/chat context from environment variables.

    Returns:
        Dict with user_id, chat_id, and speaker attribution from environment.
    """
    return {
        "user_id": os.environ.get("ASH_USER_ID"),
        "chat_id": os.environ.get("ASH_CHAT_ID"),
        "source_username": os.environ.get("ASH_USERNAME"),
        "source_display_name": os.environ.get("ASH_DISPLAY_NAME"),
    }
