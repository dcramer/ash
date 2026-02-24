"""Capability provider interfaces."""

from ash.capabilities.providers.base import (
    CapabilityAuthBeginResult,
    CapabilityAuthCompleteResult,
    CapabilityCallContext,
    CapabilityProvider,
)
from ash.capabilities.providers.subprocess import SubprocessCapabilityProvider

__all__ = [
    "CapabilityAuthBeginResult",
    "CapabilityAuthCompleteResult",
    "CapabilityCallContext",
    "CapabilityProvider",
    "SubprocessCapabilityProvider",
]
