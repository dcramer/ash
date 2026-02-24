from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ash.config import AshConfig
from ash.config.models import ModelConfig
from ash.integrations import IntegrationRuntime, create_default_integrations
from ash.integrations.rpc import active_rpc_server
from ash.integrations.runtime import IntegrationContext


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


@pytest.mark.asyncio
async def test_todo_rpc_methods_not_registered_when_todo_disabled(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AshConfig(
        workspace=tmp_path / "workspace",
        models={"default": ModelConfig(provider="openai", model="gpt-5-mini")},
    )
    components = cast(
        Any,
        SimpleNamespace(
            memory_manager=None,
            agent=cast(
                Any, SimpleNamespace(install_integration_hooks=lambda **_: None)
            ),
        ),
    )
    runtime = IntegrationRuntime(
        create_default_integrations(mode="chat", include_todo=False).contributors
    )
    context = IntegrationContext(config=config, components=components, mode="chat")
    await runtime.setup(context)

    calls: list[str] = []

    class _FakeRPCServer:
        def __init__(self, socket_path: Path) -> None:
            self.socket_path = socket_path
            self.methods: dict[str, Any] = {}

        def register(self, method: str, _handler: Any) -> None:
            calls.append(method)
            self.methods[method] = _handler

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    monkeypatch.setattr("ash.integrations.rpc.RPCServer", _FakeRPCServer)

    async with active_rpc_server(
        runtime=runtime,
        context=context,
        socket_path=tmp_path / "rpc.sock",
    ) as server:
        assert server is not None
        methods = cast(Any, server).methods
        assert not any(method.startswith("todo.") for method in methods)

    assert not any(method.startswith("todo.") for method in calls)
