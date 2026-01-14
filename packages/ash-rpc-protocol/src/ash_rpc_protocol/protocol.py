"""JSON-RPC 2.0 protocol implementation."""

import json
import struct
from dataclasses import dataclass, field
from typing import Any


# JSON-RPC 2.0 error codes
class ErrorCode:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


@dataclass
class RPCRequest:
    """JSON-RPC 2.0 request."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: int | str = 1
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "params": self.params,
            "id": self.id,
        }

    def to_bytes(self) -> bytes:
        """Serialize to length-prefixed bytes."""
        payload = json.dumps(self.to_dict()).encode()
        return struct.pack("!I", len(payload)) + payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RPCRequest":
        return cls(
            method=data.get("method", ""),
            params=data.get("params", {}),
            id=data.get("id", 1),
            jsonrpc=data.get("jsonrpc", "2.0"),
        )


@dataclass
class RPCError:
    """JSON-RPC 2.0 error."""

    code: int
    message: str
    data: Any = None

    def to_dict(self) -> dict[str, Any]:
        d = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d


@dataclass
class RPCResponse:
    """JSON-RPC 2.0 response."""

    id: int | str | None
    result: Any = None
    error: RPCError | None = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            d["error"] = self.error.to_dict()
        else:
            d["result"] = self.result
        return d

    def to_bytes(self) -> bytes:
        """Serialize to length-prefixed bytes."""
        payload = json.dumps(self.to_dict()).encode()
        return struct.pack("!I", len(payload)) + payload

    @classmethod
    def success(cls, id: int | str | None, result: Any) -> "RPCResponse":
        return cls(id=id, result=result)

    @classmethod
    def error_response(
        cls, id: int | str | None, code: int, message: str, data: Any = None
    ) -> "RPCResponse":
        return cls(id=id, error=RPCError(code=code, message=message, data=data))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RPCResponse":
        error = None
        if "error" in data:
            err = data["error"]
            error = RPCError(
                code=err.get("code", ErrorCode.INTERNAL_ERROR),
                message=err.get("message", "Unknown error"),
                data=err.get("data"),
            )
        return cls(
            id=data.get("id"),
            result=data.get("result"),
            error=error,
            jsonrpc=data.get("jsonrpc", "2.0"),
        )


async def read_message(reader) -> bytes | None:
    """Read a length-prefixed message from an async reader.

    Returns None if connection closed.
    """
    import asyncio

    try:
        length_bytes = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        return None

    length = struct.unpack("!I", length_bytes)[0]
    if length > 10 * 1024 * 1024:  # 10MB limit
        raise ValueError(f"Message too large: {length}")

    try:
        return await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        return None


def read_message_sync(sock) -> bytes | None:
    """Read a length-prefixed message from a sync socket.

    Returns None if connection closed.
    """
    length_bytes = sock.recv(4)
    if len(length_bytes) < 4:
        return None

    length = struct.unpack("!I", length_bytes)[0]
    if length > 10 * 1024 * 1024:  # 10MB limit
        raise ValueError(f"Message too large: {length}")

    # Read full message
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            return None
        data += chunk

    return data
