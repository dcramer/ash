"""Capability provider interfaces.

Spec contract: specs/capabilities.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from ash.capabilities.types import CapabilityDefinition


@dataclass(slots=True)
class CapabilityCallContext:
    """Trusted context derived from verified RPC token claims."""

    user_id: str
    chat_id: str | None
    chat_type: str | None
    provider: str | None
    thread_id: str | None
    session_key: str | None
    source_username: str | None
    source_display_name: str | None


@dataclass(slots=True)
class CapabilityAuthBeginResult:
    """Provider response for auth flow initialization."""

    auth_url: str
    expires_at: datetime | None = None
    flow_state: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CapabilityAuthCompleteResult:
    """Provider response for auth flow completion."""

    account_ref: str
    credential_material: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class CapabilityProvider(Protocol):
    """Interface for capability provider backends."""

    @property
    def namespace(self) -> str:
        """Provider namespace (prefix for capability IDs)."""

    async def definitions(self) -> list[CapabilityDefinition]:
        """Return capability definitions served by this provider."""

    async def invoke(
        self,
        *,
        capability_id: str,
        operation: str,
        input_data: dict[str, Any],
        account_ref: str | None,
        idempotency_key: str | None,
        context: CapabilityCallContext,
    ) -> dict[str, Any]:
        """Invoke an operation and return user-safe output payload."""

    async def auth_begin(
        self,
        *,
        capability_id: str,
        account_hint: str | None,
        context: CapabilityCallContext,
    ) -> CapabilityAuthBeginResult:
        """Start auth flow for a capability."""

    async def auth_complete(
        self,
        *,
        capability_id: str,
        flow_state: dict[str, Any],
        callback_url: str | None,
        code: str | None,
        context: CapabilityCallContext,
    ) -> CapabilityAuthCompleteResult:
        """Complete auth flow and return linked account result."""
