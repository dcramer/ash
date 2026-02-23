"""Shared helpers for integration composition.

Spec contract: specs/subsystems.md (Integration Hooks), specs/integrations.md.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from ash.config import AshConfig
from ash.core.prompt import PromptContext, PromptMode, RuntimeInfo
from ash.core.session import SessionState
from ash.core.types import AgentComponents
from ash.integrations.runtime import (
    IntegrationContext,
    IntegrationContributor,
    IntegrationMode,
    IntegrationRuntime,
)


def _update_subagent_shared_prompt(
    *,
    config: AshConfig,
    components: AgentComponents,
    runtime: IntegrationRuntime,
    context: IntegrationContext,
) -> None:
    tool_registry = getattr(components, "tool_registry", None)
    prompt_builder = getattr(components, "prompt_builder", None)
    if tool_registry is None or prompt_builder is None:
        return

    has_use_agent = hasattr(tool_registry, "has") and tool_registry.has("use_agent")
    has_use_skill = hasattr(tool_registry, "has") and tool_registry.has("use_skill")
    if not has_use_agent and not has_use_skill:
        return

    model = config.get_model("default")
    prompt_context = PromptContext(
        runtime=RuntimeInfo.from_environment(
            model=model.model,
            provider=model.provider,
            timezone=config.timezone,
        )
    )
    sentinel_session = SessionState(
        session_id="integration-shared-prompt",
        provider="system",
        chat_id="system",
        user_id="system",
    )
    for hook in runtime.prompt_context_augmenters(context):
        prompt_context = hook(prompt_context, sentinel_session)

    shared_prompt = prompt_builder.build(prompt_context, mode=PromptMode.MINIMAL)
    for tool_name in ("use_agent", "use_skill"):
        if not hasattr(tool_registry, "has") or not tool_registry.has(tool_name):
            continue
        tool = tool_registry.get(tool_name)
        updater = getattr(tool, "set_shared_prompt", None)
        if callable(updater):
            updater(shared_prompt)


async def compose_integrations(
    *,
    config: AshConfig,
    components: AgentComponents,
    mode: IntegrationMode,
    contributors: list[IntegrationContributor],
    sessions_path: Path | None = None,
) -> tuple[IntegrationRuntime, IntegrationContext]:
    """Build runtime/context, run setup, and install agent hooks."""
    runtime = IntegrationRuntime(contributors)
    context = IntegrationContext(
        config=config,
        components=components,
        mode=mode,
        sessions_path=sessions_path,
    )
    await runtime.setup(context)
    components.agent.install_integration_hooks(
        prompt_context_augmenters=runtime.prompt_context_augmenters(context),
        sandbox_env_augmenters=runtime.sandbox_env_augmenters(context),
        incoming_message_preprocessors=runtime.incoming_message_preprocessors(context),
        message_postprocess_hooks=runtime.message_postprocess_hooks(context),
    )
    _update_subagent_shared_prompt(
        config=config,
        components=components,
        runtime=runtime,
        context=context,
    )
    return runtime, context


@asynccontextmanager
async def active_integrations(
    *,
    config: AshConfig,
    components: AgentComponents,
    mode: IntegrationMode,
    contributors: list[IntegrationContributor],
    sessions_path: Path | None = None,
) -> AsyncIterator[tuple[IntegrationRuntime, IntegrationContext]]:
    """Compose integrations and manage startup/shutdown lifecycle."""
    runtime, context = await compose_integrations(
        config=config,
        components=components,
        mode=mode,
        contributors=contributors,
        sessions_path=sessions_path,
    )
    await runtime.on_startup(context)
    try:
        yield runtime, context
    finally:
        await runtime.on_shutdown(context)
