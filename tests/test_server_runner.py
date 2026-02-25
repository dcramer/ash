from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ash.server.runner import ServerRunner


@pytest.mark.asyncio
async def test_server_runner_without_telegram(monkeypatch) -> None:
    calls: list[str] = []

    class _FakeServer:
        should_exit = False

        async def serve(self) -> None:
            calls.append("serve")

    class _FakeLoop:
        def add_signal_handler(self, _sig, _handler) -> None:
            calls.append("signal")

        def call_soon(self, _cb) -> None:
            calls.append("call_soon")

    monkeypatch.setattr("ash.server.runner.uvicorn.Config", lambda *a, **kw: object())
    monkeypatch.setattr("ash.server.runner.uvicorn.Server", lambda _cfg: _FakeServer())
    monkeypatch.setattr(
        "ash.server.runner.asyncio.get_running_loop", lambda: _FakeLoop()
    )

    app = cast(Any, SimpleNamespace(state=SimpleNamespace(server=SimpleNamespace())))
    runner = ServerRunner(app, host="127.0.0.1", port=8080)
    await runner.run()

    assert calls.count("signal") == 2
    assert "serve" in calls
    assert "call_soon" not in calls


@pytest.mark.asyncio
async def test_server_runner_with_telegram(monkeypatch) -> None:
    calls: list[str] = []
    original_sleep = asyncio.sleep

    class _FakeServer:
        should_exit = False

        async def serve(self) -> None:
            calls.append("serve")
            for _ in range(20):
                if "telegram_start" in calls:
                    return
                await original_sleep(0)

    class _FakeLoop:
        def add_signal_handler(self, _sig, _handler) -> None:
            calls.append("signal")

        def call_soon(self, _cb) -> None:
            calls.append("call_soon")

    class _FakeProvider:
        async def start(self, cb) -> None:
            _ = cb
            calls.append("telegram_start")

        async def stop(self) -> None:
            calls.append("telegram_stop")

    handler = SimpleNamespace(handle_message=object())
    app = cast(
        Any,
        SimpleNamespace(
            state=SimpleNamespace(
                server=SimpleNamespace(get_telegram_handler=lambda: handler)
            )
        ),
    )

    async def _get_handler():
        return handler

    app.state.server.get_telegram_handler = _get_handler

    monkeypatch.setattr("ash.server.runner.uvicorn.Config", lambda *a, **kw: object())
    monkeypatch.setattr("ash.server.runner.uvicorn.Server", lambda _cfg: _FakeServer())
    monkeypatch.setattr(
        "ash.server.runner.asyncio.get_running_loop", lambda: _FakeLoop()
    )

    runner = ServerRunner(
        app,
        host="127.0.0.1",
        port=8080,
        telegram_provider=cast(Any, _FakeProvider()),
    )
    await runner.run()

    assert calls.count("signal") == 2
    assert "serve" in calls
    assert "telegram_start" in calls


@pytest.mark.asyncio
async def test_server_runner_with_telegram_handler_available_late(monkeypatch) -> None:
    calls: list[str] = []
    original_sleep = asyncio.sleep
    attempts = 0

    class _FakeServer:
        should_exit = False

        async def serve(self) -> None:
            calls.append("serve")
            for _ in range(200):
                if "telegram_start" in calls:
                    return
                await original_sleep(0)

    class _FakeLoop:
        def add_signal_handler(self, _sig, _handler) -> None:
            calls.append("signal")

        def call_soon(self, _cb) -> None:
            calls.append("call_soon")

    class _FakeProvider:
        async def start(self, cb) -> None:
            _ = cb
            calls.append("telegram_start")

        async def stop(self) -> None:
            calls.append("telegram_stop")

    handler = SimpleNamespace(handle_message=object())

    async def _get_handler():
        nonlocal attempts
        attempts += 1
        if attempts > 55:
            return handler
        return None

    async def _fast_sleep(_seconds: float) -> None:
        await original_sleep(0)

    app = cast(
        Any,
        SimpleNamespace(
            state=SimpleNamespace(
                server=SimpleNamespace(get_telegram_handler=_get_handler)
            )
        ),
    )

    monkeypatch.setattr("ash.server.runner.uvicorn.Config", lambda *a, **kw: object())
    monkeypatch.setattr("ash.server.runner.uvicorn.Server", lambda _cfg: _FakeServer())
    monkeypatch.setattr(
        "ash.server.runner.asyncio.get_running_loop", lambda: _FakeLoop()
    )
    monkeypatch.setattr("ash.server.runner.asyncio.sleep", _fast_sleep)

    runner = ServerRunner(
        app,
        host="127.0.0.1",
        port=8080,
        telegram_provider=cast(Any, _FakeProvider()),
    )
    await runner.run()

    assert attempts > 55
    assert "telegram_start" in calls
