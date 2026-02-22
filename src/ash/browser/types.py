"""Public types for browser subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

BrowserProviderName = Literal["sandbox", "kernel"]
BrowserSessionStatus = Literal["active", "closed", "archived"]
BrowserProfileStatus = Literal["active", "archived"]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class BrowserProfile:
    """Named browser profile scoped to user and provider."""

    name: str
    effective_user_id: str
    provider: BrowserProviderName
    status: BrowserProfileStatus = "active"
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class BrowserSession:
    """Browser session state."""

    id: str
    name: str
    effective_user_id: str
    provider: BrowserProviderName
    status: BrowserSessionStatus = "active"
    profile_name: str | None = None
    provider_session_id: str | None = None
    current_url: str | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class BrowserActionResult:
    """Normalized response for browser actions."""

    ok: bool
    action: str
    session_id: str | None = None
    provider: BrowserProviderName | None = None
    page_url: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    artifact_refs: list[str] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "action": self.action,
            "data": self.data,
            "artifact_refs": self.artifact_refs,
        }
        if self.session_id:
            payload["session_id"] = self.session_id
        if self.provider:
            payload["provider"] = self.provider
        if self.page_url:
            payload["page_url"] = self.page_url
        if self.error_code:
            payload["error"] = {
                "code": self.error_code,
                "message": self.error_message or "unknown_error",
            }
        return payload
