"""Browser integration contributor.

Spec contract: specs/subsystems.md (Integration Hooks).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from ash.core.prompt import PromptContext
from ash.core.prompt_keys import (
    CORE_PRINCIPLES_RULES_KEY,
    TOOL_ROUTING_RULES_KEY,
)
from ash.core.session import SessionState
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

    def augment_prompt_context(
        self,
        prompt_context: PromptContext,
        session: SessionState,
        context: IntegrationContext,
    ) -> PromptContext:
        _ = session
        if not context.config.browser.enabled:
            return prompt_context
        manager = getattr(context.components, "browser_manager", None)
        if manager is None:
            return prompt_context

        # Spec contract: specs/subsystems.md (Integration Hooks)
        # Browser-specific instruction text is contributed via structured prompt hooks.
        self._append_instruction(
            prompt_context.extra_context,
            TOOL_ROUTING_RULES_KEY,
            "Use `browser` for interactive/dynamic/authenticated workflows (click/type/wait/screenshots), or when `web_fetch` cannot access needed content.",
        )
        self._append_instruction(
            prompt_context.extra_context,
            CORE_PRINCIPLES_RULES_KEY,
            "If the user asks for a screenshot/image from browser context, run `browser` with `page.screenshot` and send the image artifact in chat via `send_message` using `image_path`.",
        )
        self._append_instruction(
            prompt_context.extra_context,
            CORE_PRINCIPLES_RULES_KEY,
            "When browser use is requested, never describe results without an actual browser tool outcome.",
        )
        return prompt_context

    def register_rpc_methods(self, server, context: IntegrationContext) -> None:
        from ash.rpc.methods.browser import register_browser_methods

        manager = getattr(context.components, "browser_manager", None)
        if manager is None:
            return
        register_browser_methods(server, manager)

    @staticmethod
    def _append_instruction(
        extra_context: dict[str, Any],
        key: str,
        line: str,
    ) -> None:
        value = extra_context.get(key)
        if isinstance(value, list):
            rules = value
        else:
            rules = []
            extra_context[key] = rules
        if line not in rules:
            rules.append(line)
