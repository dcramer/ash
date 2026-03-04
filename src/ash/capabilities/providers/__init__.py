"""Capability provider interfaces."""

from ash.capabilities.providers.base import (
    CapabilityAuthBeginResult,
    CapabilityAuthCompleteInput,
    CapabilityAuthCompleteResult,
    CapabilityAuthPollResult,
    CapabilityCallContext,
    CapabilityProvider,
)
from ash.capabilities.providers.subprocess import SubprocessCapabilityProvider

__all__ = [
    "CapabilityAuthBeginResult",
    "CapabilityAuthCompleteInput",
    "CapabilityAuthCompleteResult",
    "CapabilityAuthPollResult",
    "CapabilityCallContext",
    "CapabilityProvider",
    "SubprocessCapabilityProvider",
]
