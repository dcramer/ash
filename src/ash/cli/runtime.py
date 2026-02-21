"""Shared runtime bootstrap helpers for CLI entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ash.config import AshConfig, WorkspaceLoader
from ash.config.paths import get_graph_dir
from ash.core import AgentComponents, create_agent


@dataclass(slots=True)
class RuntimeBootstrap:
    """Composed runtime dependencies for CLI command handlers."""

    components: AgentComponents
    workspace: Any
    sentry_initialized: bool = False


async def bootstrap_runtime(
    *,
    config: AshConfig,
    model_alias: str = "default",
    initialize_sentry: bool = True,
    sentry_server_mode: bool = False,
) -> RuntimeBootstrap:
    """Initialize sentry/workspace and create a fully wired agent runtime."""
    sentry_initialized = False
    if initialize_sentry and config.sentry:
        from ash.observability import init_sentry

        sentry_initialized = init_sentry(config.sentry, server_mode=sentry_server_mode)

    workspace_loader = WorkspaceLoader(config.workspace)
    workspace_loader.ensure_workspace()
    workspace = workspace_loader.load()

    components = await create_agent(
        config=config,
        workspace=workspace,
        graph_dir=get_graph_dir(),
        model_alias=model_alias,
    )

    return RuntimeBootstrap(
        components=components,
        workspace=workspace,
        sentry_initialized=sentry_initialized,
    )
