"""Capabilities integration contributor.

Spec contract: specs/subsystems.md (Integration Hooks), specs/capabilities.md.
"""

from __future__ import annotations

from ash.core.prompt import PromptContext
from ash.core.prompt_keys import TOOL_ROUTING_RULES_KEY
from ash.core.session import SessionState
from ash.integrations.runtime import IntegrationContext, IntegrationContributor


class CapabilitiesIntegration(IntegrationContributor):
    """Owns capability manager setup and RPC surface registration."""

    name = "capabilities"
    priority = 255

    async def setup(self, context: IntegrationContext) -> None:
        from ash.capabilities import create_capability_manager

        components = context.components
        manager = getattr(components, "capability_manager", None)
        if manager is None:
            manager = await create_capability_manager()
            components.capability_manager = manager

    def register_rpc_methods(self, server, context: IntegrationContext) -> None:
        from ash.rpc.methods.capability import register_capability_methods

        manager = getattr(context.components, "capability_manager", None)
        if manager is None:
            return
        register_capability_methods(server, manager)

    def augment_prompt_context(
        self,
        prompt_context: PromptContext,
        session: SessionState,
        context: IntegrationContext,
    ) -> PromptContext:
        _ = session
        _ = context
        lines = prompt_context.extra_context.setdefault(TOOL_ROUTING_RULES_KEY, [])
        if isinstance(lines, list):
            line = (
                "- For sensitive external integrations (email/calendar), use "
                "`ash-sb capability` so identity scope is enforced by "
                "`ASH_CONTEXT_TOKEN`; do not request raw credential env vars."
            )
            if line not in lines:
                lines.append(line)
        return prompt_context
