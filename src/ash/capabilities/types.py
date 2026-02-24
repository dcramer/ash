"""Capability subsystem public types.

Spec contract: specs/capabilities.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class CapabilityOperation:
    """One operation exposed by a capability."""

    name: str
    description: str
    requires_auth: bool = True
    mutating: bool = False
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CapabilityDefinition:
    """Capability definition registered by integrations."""

    id: str
    description: str
    sensitive: bool = False
    allowed_chat_types: list[str] = field(default_factory=list)
    operations: dict[str, CapabilityOperation] = field(default_factory=dict)


@dataclass(slots=True)
class CapabilityAuthFlow:
    """Pending user-scoped auth flow for a capability."""

    flow_id: str
    capability_id: str
    user_id: str
    account_hint: str | None
    expires_at: datetime
    flow_state: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CapabilityAccount:
    """Linked account marker for a user and capability."""

    capability_id: str
    user_id: str
    account_ref: str
    created_at: datetime
    credential_material: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CapabilityInvokeResult:
    """Result returned from capability invocation."""

    request_id: str
    output: dict[str, Any]
