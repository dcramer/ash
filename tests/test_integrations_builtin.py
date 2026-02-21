from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ash.config import AshConfig
from ash.config.models import ModelConfig
from ash.integrations import (
    IntegrationContext,
    RuntimeRPCIntegration,
    SchedulingIntegration,
)
from ash.skills import SkillRegistry


def _context() -> IntegrationContext:
    config = AshConfig(
        workspace=Path("tmp-workspace"),
        models={"default": ModelConfig(provider="openai", model="gpt-5-mini")},
    )
    components = cast(
        Any,
        SimpleNamespace(
            skill_registry=SkillRegistry(),
            agent=object(),
        ),
    )
    return IntegrationContext(
        config=config,
        components=components,
        mode="serve",
    )


def test_runtime_rpc_integration_registers_config_and_logs(monkeypatch) -> None:
    context = _context()
    integration = RuntimeRPCIntegration(Path("logs"))
    server = object()
    calls: dict[str, tuple[Any, ...]] = {}

    def _register_config(s, config, skill_registry) -> None:
        calls["config"] = (s, config, skill_registry)

    def _register_logs(s, logs_path) -> None:
        calls["logs"] = (s, logs_path)

    monkeypatch.setattr(
        "ash.rpc.methods.config.register_config_methods", _register_config
    )
    monkeypatch.setattr("ash.rpc.methods.logs.register_log_methods", _register_logs)

    integration.register_rpc_methods(server, context)

    assert calls["config"][0] is server
    assert calls["config"][1] is context.config
    assert calls["config"][2] is context.components.skill_registry
    assert calls["logs"] == (server, Path("logs"))


@pytest.mark.asyncio
async def test_scheduling_integration_owns_lifecycle_and_rpc(monkeypatch) -> None:
    context = _context()
    server = object()

    class _FakeStore:
        def __init__(self, schedule_file: Path) -> None:
            self.schedule_file = schedule_file

    class _FakeWatcher:
        started = False
        stopped = False

        def __init__(self, store: Any, timezone: str) -> None:
            self.store = store
            self.timezone = timezone
            self.handlers: list[Any] = []

        def add_handler(self, handler) -> None:
            self.handlers.append(handler)

        async def start(self) -> None:
            _FakeWatcher.started = True

        async def stop(self) -> None:
            _FakeWatcher.stopped = True

    class _FakeHandler:
        def __init__(
            self,
            agent,
            senders,
            registrars,
            timezone: str,
            agent_executor,
        ) -> None:
            self.agent = agent
            self.senders = senders
            self.registrars = registrars
            self.timezone = timezone
            self.agent_executor = agent_executor

        async def handle(self, *_args, **_kwargs) -> None:
            return None

    rpc_calls: dict[str, Any] = {}

    def _register_schedule(server_obj, store_obj) -> None:
        rpc_calls["args"] = (server_obj, store_obj)

    async def _sender(
        _chat_id: str, _message: str, *, reply_to: str | None = None
    ) -> str:
        _ = reply_to
        return "msg-1"

    async def _registrar(_chat_id: str, _message_id: str) -> None:
        return None

    monkeypatch.setattr("ash.scheduling.ScheduleStore", _FakeStore)
    monkeypatch.setattr("ash.scheduling.ScheduleWatcher", _FakeWatcher)
    monkeypatch.setattr("ash.scheduling.ScheduledTaskHandler", _FakeHandler)
    monkeypatch.setattr(
        "ash.rpc.methods.schedule.register_schedule_methods", _register_schedule
    )

    integration = SchedulingIntegration(
        Path("schedule.jsonl"),
        timezone="America/Chicago",
        senders=cast(Any, {"telegram": _sender}),
        registrars=cast(Any, {"telegram": _registrar}),
    )

    await integration.setup(context)
    assert integration.store is not None
    assert integration.store.schedule_file == Path("schedule.jsonl")

    integration.register_rpc_methods(server, context)
    assert rpc_calls["args"][0] is server
    assert rpc_calls["args"][1] is integration.store

    await integration.on_startup(context)
    await integration.on_shutdown(context)
    assert _FakeWatcher.started is True
    assert _FakeWatcher.stopped is True
