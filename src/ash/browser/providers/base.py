"""Provider interface for browser actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class ProviderStartResult:
    provider_session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderGotoResult:
    url: str
    title: str | None = None
    html: str | None = None


@dataclass(slots=True)
class ProviderExtractResult:
    data: dict[str, Any]


@dataclass(slots=True)
class ProviderScreenshotResult:
    image_bytes: bytes
    mime_type: str = "image/png"


class BrowserProvider(Protocol):
    name: str

    async def start_session(
        self,
        *,
        session_id: str,
        profile_name: str | None,
        scope_key: str | None = None,
    ) -> ProviderStartResult: ...

    async def close_session(self, *, provider_session_id: str | None) -> None: ...

    async def goto(
        self,
        *,
        provider_session_id: str | None,
        url: str,
        timeout_seconds: float,
    ) -> ProviderGotoResult: ...

    async def extract(
        self,
        *,
        provider_session_id: str | None,
        html: str | None,
        mode: str,
        selector: str | None,
        max_chars: int,
    ) -> ProviderExtractResult: ...

    async def click(
        self,
        *,
        provider_session_id: str | None,
        selector: str,
    ) -> None: ...

    async def type(
        self,
        *,
        provider_session_id: str | None,
        selector: str,
        text: str,
        clear_first: bool,
    ) -> None: ...

    async def wait_for(
        self,
        *,
        provider_session_id: str | None,
        selector: str,
        timeout_seconds: float,
    ) -> None: ...

    async def screenshot(
        self,
        *,
        provider_session_id: str | None,
    ) -> ProviderScreenshotResult: ...
