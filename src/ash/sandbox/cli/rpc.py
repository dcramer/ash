"""RPC client for sandbox-to-host communication."""

import json
import os
import socket
from pathlib import Path
from typing import Any

from ash.rpc.protocol import (
    RPCRequest,
    RPCResponse,
    read_message_sync,
)

DEFAULT_SOCKET_PATH = "/run/ash/rpc.sock"


class RPCError(Exception):
    """RPC call failed."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


def rpc_call(method: str, params: dict[str, Any] | None = None) -> Any:
    """Make an RPC call to the host Ash process.

    Args:
        method: RPC method name (e.g., "memory.search").
        params: Method parameters.

    Returns:
        The result from the RPC call.

    Raises:
        RPCError: If the RPC call fails.
        ConnectionError: If unable to connect to the RPC server.
    """
    socket_path = os.environ.get("ASH_RPC_SOCKET", DEFAULT_SOCKET_PATH)

    if not Path(socket_path).exists():
        raise ConnectionError(f"RPC socket not found: {socket_path}")

    # Create request
    request = RPCRequest(method=method, params=params or {})

    # Connect and send
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

    finally:
        sock.close()


def get_context_params() -> dict[str, str | None]:
    """Get user/chat context from environment variables.

    Returns:
        Dict with user_id and chat_id from environment.
    """
    return {
        "user_id": os.environ.get("ASH_USER_ID"),
        "chat_id": os.environ.get("ASH_CHAT_ID"),
    }
