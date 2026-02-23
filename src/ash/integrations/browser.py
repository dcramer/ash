"""Browser integration contributor.

Spec contract: specs/subsystems.md (Integration Hooks).
"""

from __future__ import annotations

import asyncio
import contextlib

from ash.integrations.runtime import IntegrationContext, IntegrationContributor


class BrowserIntegration(IntegrationContributor):
    """Registers browser RPC surface when browser manager is available."""

    name = "browser"
    priority = 250

    def __init__(self) -> None:
        self._warmup_task: asyncio.Task[None] | None = None

    async def setup(self, context: IntegrationContext) -> None:
        from ash.browser import create_browser_manager
        from ash.tools.builtin import BrowserTool

        components = context.components
        manager = getattr(components, "browser_manager", None)
        if manager is None:
            manager = create_browser_manager(
                context.config,
                sandbox_executor=getattr(components, "sandbox_executor", None),
            )
            components.browser_manager = manager

        tool_registry = getattr(components, "tool_registry", None)
        if (
            context.config.browser.enabled
            and tool_registry is not None
            and hasattr(tool_registry, "has")
            and not tool_registry.has("browser")
        ):
            tool_registry.register(BrowserTool(manager))

    async def on_startup(self, context: IntegrationContext) -> None:
        manager = getattr(context.components, "browser_manager", None)
        if manager is None:
            return
        if self._warmup_task is not None and not self._warmup_task.done():
            return
        # Spec contract: specs/subsystems.md (Integration Hooks)
        # Warm browser runtime asynchronously to keep startup non-blocking.
        self._warmup_task = asyncio.create_task(
            manager.warmup_default_provider(),
            name="browser-warmup-default-provider",
        )

    async def on_shutdown(self, context: IntegrationContext) -> None:
        _ = context
        if self._warmup_task is None:
            return
        if not self._warmup_task.done():
            self._warmup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._warmup_task
        self._warmup_task = None

    def register_rpc_methods(self, server, context: IntegrationContext) -> None:
        from ash.rpc.methods.browser import register_browser_methods

        manager = getattr(context.components, "browser_manager", None)
        if manager is None:
            return
        register_browser_methods(server, manager)
