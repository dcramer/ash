"""Runtime RPC integration contributor.

Spec contract: specs/subsystems.md (Integration Hooks).
"""

from __future__ import annotations

from pathlib import Path

from ash.integrations.runtime import IntegrationContext, IntegrationContributor


class RuntimeRPCIntegration(IntegrationContributor):
    """Registers core runtime RPC surfaces."""

    name = "runtime-rpc"
    priority = 100

    def __init__(self, logs_path: Path) -> None:
        self._logs_path = logs_path

    def register_rpc_methods(self, server, context: IntegrationContext) -> None:
        from ash.rpc.methods.config import register_config_methods
        from ash.rpc.methods.logs import register_log_methods

        register_config_methods(
            server,
            context.config,
            context.components.skill_registry,
        )
        register_log_methods(server, self._logs_path)
