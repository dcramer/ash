from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ash.config import AshConfig
from ash.config.models import ModelConfig
from ash.core.prompt import PromptContext
from ash.core.session import SessionState
from ash.integrations import (
    IntegrationContext,
    IntegrationContributor,
    IntegrationRuntime,
    MemoryIntegration,
    SchedulingIntegration,
    active_integrations,
    compose_integrations,
)


class _StubContributor(IntegrationContributor):
    def __init__(
        self,
        *,
        name: str,
        priority: int,
        events: list[tuple[str, str]],
    ) -> None:
        self.name = name
        self.priority = priority
        self._events = events

    async def setup(self, context: IntegrationContext) -> None:
        self._events.append(("setup", self.name))

    async def on_startup(self, context: IntegrationContext) -> None:
        self._events.append(("startup", self.name))

    async def on_shutdown(self, context: IntegrationContext) -> None:
        self._events.append(("shutdown", self.name))

    def register_rpc_methods(self, server: Any, context: IntegrationContext) -> None:
        self._events.append(("rpc", self.name))

    def augment_prompt_context(
        self,
        prompt_context: PromptContext,
        session: SessionState,
        context: IntegrationContext,
    ) -> PromptContext:
        self._events.append(("prompt", self.name))
        prompt_context.extra_context[self.name] = True
        return prompt_context

    def augment_sandbox_env(
        self,
        env: dict[str, str],
        session: SessionState,
        effective_user_id: str,
        context: IntegrationContext,
    ) -> dict[str, str]:
        self._events.append(("env", self.name))
        env[f"HOOK_{self.name.upper()}"] = effective_user_id
        return env

    async def on_message_postprocess(
        self,
        user_message: str,
        session: SessionState,
        effective_user_id: str,
        context: IntegrationContext,
    ) -> None:
        self._events.append(("postprocess", self.name))


class _FailingContributor(_StubContributor):
    def __init__(
        self,
        *,
        name: str,
        priority: int,
        events: list[tuple[str, str]],
        fail_in: str,
    ) -> None:
        super().__init__(name=name, priority=priority, events=events)
        self._fail_in = fail_in

    async def setup(self, context: IntegrationContext) -> None:
        if self._fail_in == "setup":
            raise RuntimeError("setup failure")
        await super().setup(context)

    async def on_startup(self, context: IntegrationContext) -> None:
        if self._fail_in == "on_startup":
            raise RuntimeError("startup failure")
        await super().on_startup(context)

    async def on_shutdown(self, context: IntegrationContext) -> None:
        if self._fail_in == "on_shutdown":
            raise RuntimeError("shutdown failure")
        await super().on_shutdown(context)

    def register_rpc_methods(self, server: Any, context: IntegrationContext) -> None:
        if self._fail_in == "register_rpc_methods":
            raise RuntimeError("rpc failure")
        super().register_rpc_methods(server, context)

    def augment_prompt_context(
        self,
        prompt_context: PromptContext,
        session: SessionState,
        context: IntegrationContext,
    ) -> PromptContext:
        if self._fail_in == "augment_prompt_context":
            raise RuntimeError("prompt failure")
        return super().augment_prompt_context(prompt_context, session, context)

    def augment_sandbox_env(
        self,
        env: dict[str, str],
        session: SessionState,
        effective_user_id: str,
        context: IntegrationContext,
    ) -> dict[str, str]:
        if self._fail_in == "augment_sandbox_env":
            raise RuntimeError("env failure")
        return super().augment_sandbox_env(env, session, effective_user_id, context)

    async def on_message_postprocess(
        self,
        user_message: str,
        session: SessionState,
        effective_user_id: str,
        context: IntegrationContext,
    ) -> None:
        if self._fail_in == "on_message_postprocess":
            raise RuntimeError("postprocess failure")
        await super().on_message_postprocess(
            user_message, session, effective_user_id, context
        )


def _context() -> IntegrationContext:
    config = AshConfig(
        workspace=Path("tmp-workspace"),
        models={"default": ModelConfig(provider="openai", model="gpt-5-mini")},
    )
    components = cast(Any, object())  # not used in these runtime tests
    return IntegrationContext(config=config, components=components, mode="eval")


@pytest.mark.asyncio
async def test_integration_runtime_runs_in_deterministic_order() -> None:
    events: list[tuple[str, str]] = []
    runtime = IntegrationRuntime(
        [
            _StubContributor(name="b", priority=200, events=events),
            _StubContributor(name="a", priority=200, events=events),
            _StubContributor(name="z", priority=100, events=events),
        ]
    )
    context = _context()

    await runtime.setup(context)
    await runtime.on_startup(context)
    runtime.register_rpc_methods(cast(Any, object()), context)
    await runtime.on_shutdown(context)

    assert events == [
        ("setup", "z"),
        ("setup", "a"),
        ("setup", "b"),
        ("startup", "z"),
        ("startup", "a"),
        ("startup", "b"),
        ("rpc", "z"),
        ("rpc", "a"),
        ("rpc", "b"),
        ("shutdown", "b"),
        ("shutdown", "a"),
        ("shutdown", "z"),
    ]


def test_integration_runtime_builds_prompt_and_env_hooks() -> None:
    events: list[tuple[str, str]] = []
    runtime = IntegrationRuntime(
        [
            _StubContributor(name="a", priority=10, events=events),
            _StubContributor(name="b", priority=20, events=events),
        ]
    )
    context = _context()
    session = SessionState(
        session_id="s-1",
        provider="telegram",
        chat_id="c-1",
        user_id="u-1",
    )

    prompt_context = PromptContext()
    for hook in runtime.prompt_context_augmenters(context):
        prompt_context = hook(prompt_context, session)

    env = {}
    for hook in runtime.sandbox_env_augmenters(context):
        env = hook(env, session, "user-123")

    assert prompt_context.extra_context == {"a": True, "b": True}
    assert env == {"HOOK_A": "user-123", "HOOK_B": "user-123"}
    assert events == [
        ("prompt", "a"),
        ("prompt", "b"),
        ("env", "a"),
        ("env", "b"),
    ]


@pytest.mark.asyncio
async def test_integration_runtime_builds_postprocess_hooks() -> None:
    events: list[tuple[str, str]] = []
    runtime = IntegrationRuntime(
        [
            _StubContributor(name="a", priority=10, events=events),
            _StubContributor(name="b", priority=20, events=events),
        ]
    )
    context = _context()
    session = SessionState(
        session_id="s-1",
        provider="telegram",
        chat_id="c-1",
        user_id="u-1",
    )

    for hook in runtime.message_postprocess_hooks(context):
        await hook("remember this", session, "user-123")

    assert events == [
        ("postprocess", "a"),
        ("postprocess", "b"),
    ]


@pytest.mark.asyncio
async def test_integration_runtime_setup_failure_disables_only_failing_contributor() -> (
    None
):
    events: list[tuple[str, str]] = []
    runtime = IntegrationRuntime(
        [
            _StubContributor(name="ok", priority=10, events=events),
            _FailingContributor(
                name="bad",
                priority=20,
                events=events,
                fail_in="setup",
            ),
        ]
    )
    context = _context()

    await runtime.setup(context)
    await runtime.on_startup(context)
    runtime.register_rpc_methods(cast(Any, object()), context)
    await runtime.on_shutdown(context)

    assert [contributor.name for contributor in runtime.active_contributors] == ["ok"]
    assert events == [
        ("setup", "ok"),
        ("startup", "ok"),
        ("rpc", "ok"),
        ("shutdown", "ok"),
    ]


@pytest.mark.asyncio
async def test_integration_runtime_isolates_hook_failures_after_setup() -> None:
    events: list[tuple[str, str]] = []
    runtime = IntegrationRuntime(
        [
            _FailingContributor(
                name="bad",
                priority=10,
                events=events,
                fail_in="on_message_postprocess",
            ),
            _StubContributor(name="ok", priority=20, events=events),
        ]
    )
    context = _context()
    session = SessionState(
        session_id="s-1",
        provider="telegram",
        chat_id="c-1",
        user_id="u-1",
    )

    await runtime.setup(context)
    for hook in runtime.prompt_context_augmenters(context):
        _ = hook(PromptContext(), session)
    for hook in runtime.sandbox_env_augmenters(context):
        _ = hook({}, session, "user-123")
    runtime.register_rpc_methods(cast(Any, object()), context)
    await runtime.on_startup(context)
    for hook in runtime.message_postprocess_hooks(context):
        await hook("remember this", session, "user-123")
    await runtime.on_shutdown(context)

    assert ("postprocess", "ok") in events


@pytest.mark.asyncio
async def test_compose_integrations_runs_setup_and_installs_hooks() -> None:
    events: list[tuple[str, str]] = []

    class _FakeAgent:
        def __init__(self) -> None:
            self.prompt_hooks = None
            self.env_hooks = None
            self.postprocess_hooks = None

        def install_integration_hooks(
            self,
            *,
            prompt_context_augmenters=None,
            sandbox_env_augmenters=None,
            message_postprocess_hooks=None,
        ) -> None:
            self.prompt_hooks = prompt_context_augmenters
            self.env_hooks = sandbox_env_augmenters
            self.postprocess_hooks = message_postprocess_hooks

    config = AshConfig(
        workspace=Path("tmp-workspace"),
        models={"default": ModelConfig(provider="openai", model="gpt-5-mini")},
    )
    fake_agent = _FakeAgent()
    components = cast(Any, SimpleNamespace(agent=fake_agent))
    runtime, context = await compose_integrations(
        config=config,
        components=components,
        mode="eval",
        contributors=[_StubContributor(name="x", priority=10, events=events)],
    )

    assert isinstance(runtime, IntegrationRuntime)
    assert context.mode == "eval"
    assert events == [("setup", "x")]
    assert fake_agent.prompt_hooks is not None
    assert fake_agent.env_hooks is not None
    assert fake_agent.postprocess_hooks is not None
    assert len(fake_agent.prompt_hooks) == 1
    assert len(fake_agent.env_hooks) == 1
    assert len(fake_agent.postprocess_hooks) == 1


@pytest.mark.asyncio
async def test_active_integrations_runs_full_lifecycle() -> None:
    events: list[tuple[str, str]] = []

    class _FakeAgent:
        def install_integration_hooks(
            self,
            *,
            prompt_context_augmenters=None,
            sandbox_env_augmenters=None,
            message_postprocess_hooks=None,
        ) -> None:
            _ = (
                prompt_context_augmenters,
                sandbox_env_augmenters,
                message_postprocess_hooks,
            )

    config = AshConfig(
        workspace=Path("tmp-workspace"),
        models={"default": ModelConfig(provider="openai", model="gpt-5-mini")},
    )
    components = cast(Any, SimpleNamespace(agent=_FakeAgent()))

    async with active_integrations(
        config=config,
        components=components,
        mode="eval",
        contributors=[_StubContributor(name="x", priority=10, events=events)],
    ):
        events.append(("inside", "ok"))

    assert events == [
        ("setup", "x"),
        ("startup", "x"),
        ("inside", "ok"),
        ("shutdown", "x"),
    ]


@pytest.mark.asyncio
async def test_memory_and_scheduling_compose_with_single_memory_postprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []

    class _FakeAgent:
        def __init__(self) -> None:
            self.postprocess_hooks = None

        def install_integration_hooks(
            self,
            *,
            prompt_context_augmenters=None,
            sandbox_env_augmenters=None,
            message_postprocess_hooks=None,
        ) -> None:
            _ = (prompt_context_augmenters, sandbox_env_augmenters)
            self.postprocess_hooks = message_postprocess_hooks

    class _FakeMemoryPostprocessService:
        def __init__(
            self,
            *,
            store: object | None,
            extractor: object | None,
            extraction_enabled: bool,
            min_message_length: int,
            debounce_seconds: int,
            confidence_threshold: float,
        ) -> None:
            _ = (
                store,
                extractor,
                extraction_enabled,
                min_message_length,
                debounce_seconds,
                confidence_threshold,
            )
            events.append(("memory_postprocess_init", "ok"))

        def maybe_schedule(
            self,
            *,
            user_message: str,
            session: SessionState,
            effective_user_id: str,
        ) -> None:
            _ = (user_message, session)
            events.append(("memory_postprocess", effective_user_id))

    monkeypatch.setattr(
        "ash.memory.postprocess.MemoryPostprocessService",
        _FakeMemoryPostprocessService,
    )

    config = AshConfig(
        workspace=tmp_path / "workspace",
        models={"default": ModelConfig(provider="openai", model="gpt-5-mini")},
    )
    fake_agent = _FakeAgent()
    components = cast(
        Any,
        SimpleNamespace(
            agent=fake_agent,
            memory_manager=object(),
            memory_extractor=None,
        ),
    )

    runtime, context = await compose_integrations(
        config=config,
        components=components,
        mode="eval",
        contributors=[
            SchedulingIntegration(tmp_path / "schedule.jsonl"),
            MemoryIntegration(),
        ],
    )

    # Memory runs before scheduling by priority.
    assert [c.name for c in runtime.contributors] == ["memory", "scheduling"]
    assert fake_agent.postprocess_hooks is not None
    assert len(fake_agent.postprocess_hooks) == 2

    session = SessionState(
        session_id="s-1",
        provider="telegram",
        chat_id="c-1",
        user_id="u-1",
    )
    for hook in runtime.message_postprocess_hooks(context):
        await hook("remember this", session, "user-123")

    # Only memory integration should produce postprocess side effects.
    assert events == [
        ("memory_postprocess_init", "ok"),
        ("memory_postprocess", "user-123"),
    ]
