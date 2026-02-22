"""Memory integration contributor.

Spec contract: specs/subsystems.md (Integration Hooks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ash.integrations.runtime import IntegrationContext, IntegrationContributor

if TYPE_CHECKING:
    from ash.core.session import SessionState
    from ash.memory.postprocess import MemoryPostprocessService


class MemoryIntegration(IntegrationContributor):
    """Registers memory RPC surface when memory is enabled."""

    name = "memory"
    priority = 200

    def __init__(self) -> None:
        self._postprocess: MemoryPostprocessService | None = None

    async def setup(self, context: IntegrationContext) -> None:
        from ash.memory.postprocess import MemoryPostprocessService

        components = context.components
        if not components.memory_manager:
            return

        memory_config = context.config.memory
        self._postprocess = MemoryPostprocessService(
            store=components.memory_manager,
            extractor=components.memory_extractor,
            extraction_enabled=memory_config.extraction_enabled,
            min_message_length=memory_config.extraction_min_message_length,
            debounce_seconds=memory_config.extraction_debounce_seconds,
            context_messages=memory_config.extraction_context_messages,
            confidence_threshold=memory_config.extraction_confidence_threshold,
        )

    def register_rpc_methods(self, server, context: IntegrationContext) -> None:
        from ash.rpc.methods.memory import register_memory_methods

        components = context.components
        if not components.memory_manager:
            return
        register_memory_methods(
            server,
            components.memory_manager,
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
        if self._postprocess is None:
            return
        self._postprocess.maybe_schedule(
            user_message=user_message,
            session=session,
            effective_user_id=effective_user_id,
        )
