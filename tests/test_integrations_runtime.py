from __future__ import annotations

from pathlib import Path
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
