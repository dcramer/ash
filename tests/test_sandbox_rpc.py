"""Tests for ash-sb RPC transport behavior."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path
from typing import Any

from ash_rpc_protocol import RPCRequest, RPCResponse, read_message_sync
from ash_sandbox_cli.rpc import rpc_call


def _serve_once(
    sock: socket.socket, result_payload: dict[str, Any]
) -> threading.Thread:
    """Serve exactly one framed RPC request and then exit."""

    def _worker() -> None:
        conn, _ = sock.accept()
        try:
            raw = read_message_sync(conn)
            assert raw is not None
            request = RPCRequest.from_dict(json.loads(raw))
            response = RPCResponse.success(request.id, result_payload)
            conn.sendall(response.to_bytes())
        finally:
            conn.close()
            sock.close()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def test_rpc_call_uses_unix_socket_transport(
    tmp_path: Path,
    monkeypatch,
) -> None:
    socket_path = tmp_path / "rpc.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)
    thread = _serve_once(server, {"transport": "unix"})

    monkeypatch.setenv("ASH_RPC_SOCKET", str(socket_path))
    monkeypatch.setenv("ASH_CONTEXT_TOKEN", "a.b.c")
    monkeypatch.delenv("ASH_RPC_HOST", raising=False)
    monkeypatch.delenv("ASH_RPC_PORT", raising=False)

    result = rpc_call("memory.search", {"query": "pizza"}, max_retries=0)
    thread.join(timeout=2)

    assert result == {"transport": "unix"}


def test_rpc_call_falls_back_to_tcp_when_unix_connect_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Create a stale Unix socket path (file exists, no listener), which yields
    # connection-refused behavior on connect().
    stale_path = tmp_path / "stale.sock"
    stale_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale_server.bind(str(stale_path))
    stale_server.close()

    tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_server.bind(("127.0.0.1", 0))
    tcp_server.listen(1)
    tcp_port = int(tcp_server.getsockname()[1])
    thread = _serve_once(tcp_server, {"transport": "tcp"})

    monkeypatch.setenv("ASH_RPC_SOCKET", str(stale_path))
    monkeypatch.setenv("ASH_RPC_HOST", "127.0.0.1")
    monkeypatch.setenv("ASH_RPC_PORT", str(tcp_port))
    monkeypatch.setenv("ASH_CONTEXT_TOKEN", "a.b.c")

    result = rpc_call("memory.search", {"query": "pizza"}, max_retries=0)
    thread.join(timeout=2)

    assert result == {"transport": "tcp"}
