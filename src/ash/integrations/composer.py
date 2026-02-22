"""Shared helpers for integration composition."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from ash.config import AshConfig
from ash.core.types import AgentComponents
from ash.integrations.runtime import (
    IntegrationContext,
    IntegrationContributor,
    IntegrationRuntime,
)


async def compose_integrations(
    *,
    config: AshConfig,
    components: AgentComponents,
    mode: Literal["serve", "chat", "eval"],
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
        message_postprocess_hooks=runtime.message_postprocess_hooks(context),
    )
    return runtime, context


@asynccontextmanager
async def active_integrations(
    *,
    config: AshConfig,
    components: AgentComponents,
    mode: Literal["serve", "chat", "eval"],
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
