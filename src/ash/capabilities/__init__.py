"""Capability subsystem public API.

Public API:
- CapabilityManager: Main entry point
- create_capability_manager: Factory function

Types:
- CapabilityDefinition
- CapabilityOperation
- CapabilityAuthFlow
- CapabilityAccount
- CapabilityInvokeResult

Spec contract: specs/capabilities.md.
"""

from ash.capabilities.manager import (
    CapabilityError,
    CapabilityManager,
    create_capability_manager,
)
from ash.capabilities.providers import (
    CapabilityAuthBeginResult,
    CapabilityAuthCompleteResult,
    CapabilityCallContext,
    CapabilityProvider,
    SubprocessCapabilityProvider,
)
from ash.capabilities.types import (
    CapabilityAccount,
    CapabilityAuthFlow,
    CapabilityDefinition,
    CapabilityInvokeResult,
    CapabilityOperation,
)

__all__ = [
    "CapabilityError",
    "CapabilityManager",
    "CapabilityProvider",
    "CapabilityCallContext",
    "CapabilityAuthBeginResult",
    "CapabilityAuthCompleteResult",
    "CapabilityAccount",
    "CapabilityAuthFlow",
    "CapabilityDefinition",
    "CapabilityInvokeResult",
    "CapabilityOperation",
    "SubprocessCapabilityProvider",
    "create_capability_manager",
]
