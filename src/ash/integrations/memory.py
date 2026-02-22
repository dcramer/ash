"""Memory integration contributor.

Spec contract: specs/subsystems.md (Integration Hooks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ash.integrations.runtime import IntegrationContext, IntegrationContributor

if TYPE_CHECKING:
    from ash.core.session import SessionState


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

    async def on_message_postprocess(
        self,
        user_message: str,
        session: SessionState,
        effective_user_id: str,
        context: IntegrationContext,
    ) -> None:
        context.components.agent.run_memory_postprocess(
            user_message=user_message,
            session=session,
            effective_user_id=effective_user_id,
        )
