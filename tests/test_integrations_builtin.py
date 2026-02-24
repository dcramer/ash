from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ash.config import AshConfig
from ash.config.models import ModelConfig
from ash.core.prompt import PromptContext
from ash.core.session import SessionState
from ash.integrations import (
    BrowserIntegration,
    CapabilitiesIntegration,
    IntegrationContext,
    RuntimeRPCIntegration,
    SchedulingIntegration,
    TodoIntegration,
)
from ash.skills import SkillRegistry
from ash.tools import ToolRegistry


def _context() -> IntegrationContext:
    config = AshConfig(
        workspace=Path("tmp-workspace"),
        models={"default": ModelConfig(provider="openai", model="gpt-5-mini")},
    )
    components = cast(
        Any,
        SimpleNamespace(
            skill_registry=SkillRegistry(),
            tool_registry=ToolRegistry(),
            sandbox_executor=None,
            browser_manager=None,
            memory_manager=None,
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
        def __init__(self, graph_dir: Path) -> None:
            self.graph_dir = graph_dir

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
            persisters,
            timezone: str,
            agent_executor,
        ) -> None:
            self.agent = agent
            self.senders = senders
            self.registrars = registrars
            self.persisters = persisters
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
        Path("graph"),
        timezone="America/Chicago",
        senders=cast(Any, {"telegram": _sender}),
        registrars=cast(Any, {"telegram": _registrar}),
    )

    await integration.setup(context)
    assert integration.store is not None
    assert integration.store.graph_dir == Path("graph")

    integration.register_rpc_methods(server, context)
    assert rpc_calls["args"][0] is server
    assert rpc_calls["args"][1] is integration.store

    await integration.on_startup(context)
    await integration.on_shutdown(context)
    assert _FakeWatcher.started is True
    assert _FakeWatcher.stopped is True


@pytest.mark.asyncio
async def test_browser_integration_owns_manager_tool_and_warmup(monkeypatch) -> None:
    context = _context()
    integration = BrowserIntegration()

    class _FakeManager:
        def __init__(self) -> None:
            self.warmup_calls = 0
            self.shutdown_calls = 0

        async def warmup_default_provider(self) -> None:
            self.warmup_calls += 1

        async def shutdown(self) -> None:
            self.shutdown_calls += 1

    fake_manager = _FakeManager()
    monkeypatch.setattr(
        "ash.browser.create_browser_manager",
        lambda *args, **kwargs: fake_manager,
    )

    await integration.setup(context)
    assert context.components.browser_manager is fake_manager
    assert context.components.tool_registry.has("browser")

    await integration.on_startup(context)
    await asyncio.sleep(0)
    assert fake_manager.warmup_calls == 1
    await integration.on_shutdown(context)
    assert fake_manager.shutdown_calls == 1


@pytest.mark.asyncio
async def test_browser_integration_injects_prompt_guidance_via_hook(
    monkeypatch,
) -> None:
    context = _context()
    integration = BrowserIntegration()

    class _FakeManager:
        async def warmup_default_provider(self) -> None:
            return None

    fake_manager = _FakeManager()
    monkeypatch.setattr(
        "ash.browser.create_browser_manager",
        lambda *args, **kwargs: fake_manager,
    )

    # Drive setup so manager exists on context components.
    await integration.setup(context)

    session = SessionState(
        session_id="s-1",
        provider="telegram",
        chat_id="c-1",
        user_id="u-1",
    )
    prompt_context = integration.augment_prompt_context(
        PromptContext(),
        session,
        context,
    )
    routing = prompt_context.extra_context.get("tool_routing_rules")
    principles = prompt_context.extra_context.get("core_principles_rules")
    assert isinstance(routing, list)
    assert isinstance(principles, list)
    assert any("Use `browser` for interactive" in line for line in routing)
    assert any("page.screenshot" in line for line in principles)


@pytest.mark.asyncio
async def test_todo_integration_registers_todo_rpc_methods(monkeypatch) -> None:
    context = _context()
    server = object()

    calls: dict[str, Any] = {}

    def _register_todo(server_obj, manager_obj, schedule_store=None) -> None:
        calls["args"] = (server_obj, manager_obj, schedule_store)

    monkeypatch.setattr("ash.rpc.methods.todo.register_todo_methods", _register_todo)

    integration = TodoIntegration(
        graph_dir=Path("graph"),
        schedule_enabled=True,
    )
    await integration.setup(context)
    assert integration.manager is not None

    integration.register_rpc_methods(server, context)
    assert calls["args"][0] is server
    assert calls["args"][1] is integration.manager
    assert calls["args"][2] is not None


@pytest.mark.asyncio
async def test_todo_integration_owns_prompt_routing_guidance(monkeypatch) -> None:
    context = _context()
    integration = TodoIntegration(graph_dir=Path("graph"))

    monkeypatch.setattr(
        "ash.integrations.todo.create_todo_manager",
        lambda *args, **kwargs: asyncio.sleep(0, result=cast(Any, object())),
    )

    await integration.setup(context)
    session = SessionState(
        session_id="s-1",
        provider="telegram",
        chat_id="c-1",
        user_id="u-1",
    )
    prompt_context = integration.augment_prompt_context(
        PromptContext(), session, context
    )
    routing = prompt_context.extra_context.get("tool_routing_rules")
    assert isinstance(routing, list)
    assert any("ash-sb todo add" in line for line in routing)


@pytest.mark.asyncio
async def test_capabilities_integration_registers_capability_rpc_methods(
    monkeypatch,
) -> None:
    context = _context()
    server = object()
    calls: dict[str, Any] = {}

    def _register_capability(server_obj, manager_obj) -> None:
        calls["args"] = (server_obj, manager_obj)

    monkeypatch.setattr(
        "ash.rpc.methods.capability.register_capability_methods",
        _register_capability,
    )

    integration = CapabilitiesIntegration()
    await integration.setup(context)
    assert getattr(context.components, "capability_manager", None) is not None

    integration.register_rpc_methods(server, context)
    assert calls["args"][0] is server
    assert calls["args"][1] is context.components.capability_manager


@pytest.mark.asyncio
async def test_capabilities_integration_owns_prompt_routing_guidance() -> None:
    context = _context()
    integration = CapabilitiesIntegration()
    await integration.setup(context)
    session = SessionState(
        session_id="s-1",
        provider="telegram",
        chat_id="c-1",
        user_id="u-1",
    )
    prompt_context = integration.augment_prompt_context(
        PromptContext(),
        session,
        context,
    )
    routing = prompt_context.extra_context.get("tool_routing_rules")
    assert isinstance(routing, list)
    assert any("ash-sb capability" in line for line in routing)
