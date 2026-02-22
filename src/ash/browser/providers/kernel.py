"""Kernel remote browser provider adapter."""

from __future__ import annotations

import asyncio
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ash.browser.providers.base import (
    ProviderExtractResult,
    ProviderGotoResult,
    ProviderScreenshotResult,
    ProviderStartResult,
)


class KernelBrowserProvider:
    """Kernel provider adapter.

    This adapter intentionally starts minimal and supports deterministic errors for
    actions that need deeper endpoint-specific wiring.
    """

    name = "kernel"

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        project_id: str | None,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._base_url = base_url.rstrip("/")
        self._project_id = project_id

    def _auth_headers(self) -> dict[str, str]:
        if not self._api_key:
            raise ValueError("kernel_api_key_missing")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._project_id:
            headers["X-Project-Id"] = self._project_id
        return headers

    def _blocking_start_session(self, *, session_id: str) -> None:
        req = Request(  # noqa: S310
            f"{self._base_url}/sessions",
            method="POST",
            data=(f'{{"client_session_id":"{session_id}"}}').encode(),
            headers=self._auth_headers(),
        )
        with urlopen(req, timeout=10):  # noqa: S310
            pass

    def _blocking_close_session(self, *, provider_session_id: str) -> None:
        req = Request(  # noqa: S310
            f"{self._base_url}/sessions/{provider_session_id}",
            method="DELETE",
            headers=self._auth_headers(),
        )
        with urlopen(req, timeout=10):  # noqa: S310
            pass

    async def start_session(
        self,
        *,
        session_id: str,
        profile_name: str | None,
    ) -> ProviderStartResult:
        _ = profile_name
        try:
            await asyncio.to_thread(self._blocking_start_session, session_id=session_id)
        except HTTPError as e:
            raise ValueError(f"kernel_start_http_{e.code}") from None
        except Exception as e:
            raise ValueError(f"kernel_start_failed:{e}") from e
        return ProviderStartResult(provider_session_id=session_id)

    async def close_session(self, *, provider_session_id: str | None) -> None:
        if not provider_session_id:
            return
        try:
            await asyncio.to_thread(
                self._blocking_close_session,
                provider_session_id=provider_session_id,
            )
        except Exception:
            return None

    async def goto(
        self,
        *,
        provider_session_id: str | None,
        url: str,
        timeout_seconds: float,
    ) -> ProviderGotoResult:
        _ = (provider_session_id, timeout_seconds)
        raise ValueError("kernel_action_not_implemented")

    async def extract(
        self,
        *,
        provider_session_id: str | None,
        html: str | None,
        mode: str,
        selector: str | None,
        max_chars: int,
    ) -> ProviderExtractResult:
        _ = (provider_session_id, html, mode, selector, max_chars)
        raise ValueError("kernel_action_not_implemented")

    async def click(
        self,
        *,
        provider_session_id: str | None,
        selector: str,
    ) -> None:
        _ = (provider_session_id, selector)
        raise ValueError("kernel_action_not_implemented")

    async def type(
        self,
        *,
        provider_session_id: str | None,
        selector: str,
        text: str,
        clear_first: bool,
    ) -> None:
        _ = (provider_session_id, selector, text, clear_first)
        raise ValueError("kernel_action_not_implemented")

    async def wait_for(
        self,
        *,
        provider_session_id: str | None,
        selector: str,
        timeout_seconds: float,
    ) -> None:
        _ = (provider_session_id, selector, timeout_seconds)
        raise ValueError("kernel_action_not_implemented")

    async def screenshot(
        self,
        *,
        provider_session_id: str | None,
    ) -> ProviderScreenshotResult:
        _ = provider_session_id
        raise ValueError("kernel_action_not_implemented")
