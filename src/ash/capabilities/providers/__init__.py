"""Capability provider interfaces."""

from ash.capabilities.providers.base import (
    CapabilityAuthBeginResult,
    CapabilityAuthCompleteResult,
    CapabilityAuthPollResult,
    CapabilityCallContext,
    CapabilityProvider,
)
from ash.capabilities.providers.subprocess import SubprocessCapabilityProvider

__all__ = [
    "CapabilityAuthBeginResult",
    "CapabilityAuthCompleteResult",
    "CapabilityAuthPollResult",
    "CapabilityCallContext",
    "CapabilityProvider",
    "SubprocessCapabilityProvider",
]
