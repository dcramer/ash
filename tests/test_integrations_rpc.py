from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ash.integrations.rpc import active_rpc_server


@pytest.mark.asyncio
async def test_active_rpc_server_starts_and_stops(monkeypatch) -> None:
    events: list[str] = []

    class _FakeRPCServer:
        def __init__(self, socket_path: Path) -> None:
            self.socket_path = socket_path
            events.append("init")

        async def start(self) -> None:
            events.append("start")

        async def stop(self) -> None:
            events.append("stop")

    runtime = cast(
        Any,
        SimpleNamespace(
            register_rpc_methods=lambda _server, _context: events.append("register")
        ),
    )
    context = cast(Any, object())

    monkeypatch.setattr("ash.integrations.rpc.RPCServer", _FakeRPCServer)

    async with active_rpc_server(
        runtime=runtime,
        context=context,
        socket_path=Path("rpc.sock"),
    ) as server:
        assert server is not None
        assert server.socket_path == Path("rpc.sock")
        events.append("inside")

    assert events == ["init", "register", "start", "inside", "stop"]


@pytest.mark.asyncio
async def test_active_rpc_server_noops_when_disabled(monkeypatch) -> None:
    class _FakeRPCServer:
        def __init__(self, socket_path: Path) -> None:
            raise AssertionError("RPC server should not be constructed")

    runtime = cast(
        Any,
        SimpleNamespace(
            register_rpc_methods=lambda _server, _context: None,
        ),
    )
    context = cast(Any, object())

    monkeypatch.setattr("ash.integrations.rpc.RPCServer", _FakeRPCServer)

    async with active_rpc_server(
        runtime=runtime,
        context=context,
        socket_path=Path("rpc.sock"),
        enabled=False,
    ) as server:
        assert server is None
