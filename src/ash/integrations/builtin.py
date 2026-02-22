"""Compatibility exports for built-in integration contributors.

Prefer importing from dedicated modules:
- ash.integrations.memory
- ash.integrations.scheduling
- ash.integrations.runtime_rpc
"""

from ash.integrations.memory import MemoryIntegration
from ash.integrations.runtime_rpc import RuntimeRPCIntegration
from ash.integrations.scheduling import SchedulingIntegration

__all__ = [
    "MemoryIntegration",
    "RuntimeRPCIntegration",
    "SchedulingIntegration",
]
