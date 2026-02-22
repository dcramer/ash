"""Memory integration contributor.

Spec contract: specs/subsystems.md (Integration Hooks).
"""

from __future__ import annotations

from ash.integrations.runtime import IntegrationContext, IntegrationContributor


class MemoryIntegration(IntegrationContributor):
    """Registers memory RPC surface when memory is enabled."""

    name = "memory"
    priority = 200

    def register_rpc_methods(self, server, context: IntegrationContext) -> None:
        from ash.rpc.methods.memory import register_memory_methods

        components = context.components
        if not components.memory_manager:
            return
        register_memory_methods(
            server,
            components.memory_manager,
            components.person_manager,
            memory_extractor=components.memory_extractor,
            sessions_path=context.sessions_path,
        )
