"""Capability provider interfaces."""

from ash.capabilities.providers.base import (
    CapabilityAuthBeginResult,
    CapabilityAuthCompleteResult,
    CapabilityCallContext,
    CapabilityProvider,
)

__all__ = [
    "CapabilityAuthBeginResult",
    "CapabilityAuthCompleteResult",
    "CapabilityCallContext",
    "CapabilityProvider",
]
