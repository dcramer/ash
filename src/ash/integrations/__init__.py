"""Integration hooks runtime.

Spec contract: specs/subsystems.md (Integration Hooks).
"""

from ash.integrations.composer import active_integrations, compose_integrations
from ash.integrations.defaults import DefaultIntegrations, create_default_integrations
from ash.integrations.memory import MemoryIntegration
from ash.integrations.rpc import active_rpc_server
from ash.integrations.runtime import (
    IntegrationContext,
    IntegrationContributor,
    IntegrationRuntime,
)
from ash.integrations.runtime_rpc import RuntimeRPCIntegration
from ash.integrations.scheduling import SchedulingIntegration

__all__ = [
    "MemoryIntegration",
    "RuntimeRPCIntegration",
    "SchedulingIntegration",
    "compose_integrations",
    "active_integrations",
    "DefaultIntegrations",
    "create_default_integrations",
    "active_rpc_server",
    "IntegrationContext",
    "IntegrationContributor",
    "IntegrationRuntime",
]
