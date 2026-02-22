"""Browser integration contributor.

Spec contract: specs/subsystems.md (Integration Hooks).
"""

from __future__ import annotations

from ash.integrations.runtime import IntegrationContext, IntegrationContributor


class BrowserIntegration(IntegrationContributor):
    """Registers browser RPC surface when browser manager is available."""

    name = "browser"
    priority = 250

    def register_rpc_methods(self, server, context: IntegrationContext) -> None:
        from ash.rpc.methods.browser import register_browser_methods

        manager = context.components.browser_manager
        if manager is None:
            return
        register_browser_methods(server, manager)
