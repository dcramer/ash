"""Default sandbox-backed browser provider.

This v1 implementation uses HTTP fetching as deterministic fallback for browsing
workflows when full headless browser automation is unavailable.
"""

from __future__ import annotations

import asyncio
import re
from html import unescape
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ash.browser.providers.base import (
    ProviderExtractResult,
    ProviderGotoResult,
    ProviderScreenshotResult,
    ProviderStartResult,
)


class SandboxBrowserProvider:
    """Sandbox browser provider using deterministic HTTP + HTML extraction."""

    name = "sandbox"

    @staticmethod
    def _blocking_fetch(url: str, timeout_seconds: float) -> tuple[str, str]:
        headers = {
            "User-Agent": "AshBrowser/1.0 (+https://github.com/dcramer/ash)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        req = Request(url, headers=headers)  # noqa: S310
        with urlopen(req, timeout=max(1.0, timeout_seconds)) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", errors="replace")
            final_url = resp.geturl()
        return final_url, body

    async def start_session(
        self,
        *,
        session_id: str,
        profile_name: str | None,
    ) -> ProviderStartResult:
        _ = (session_id, profile_name)
        return ProviderStartResult(provider_session_id=None, metadata={})

    async def close_session(self, *, provider_session_id: str | None) -> None:
        _ = provider_session_id
        return None

    async def goto(
        self,
        *,
        provider_session_id: str | None,
        url: str,
        timeout_seconds: float,
    ) -> ProviderGotoResult:
        _ = provider_session_id
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("invalid_url_scheme")
        final_url, body = await asyncio.to_thread(
            self._blocking_fetch,
            url,
            timeout_seconds,
        )

        title_match = re.search(
            r"<title[^>]*>(.*?)</title>", body, re.IGNORECASE | re.DOTALL
        )
        title = unescape(title_match.group(1).strip()) if title_match else None
        return ProviderGotoResult(url=final_url, title=title, html=body)

    async def extract(
        self,
        *,
        provider_session_id: str | None,
        html: str | None,
        mode: str,
        selector: str | None,
        max_chars: int,
    ) -> ProviderExtractResult:
        _ = (provider_session_id, selector)
        if not html:
            return ProviderExtractResult(data={"text": ""})

        if mode not in {"text", "title"}:
            raise ValueError("unsupported_extract_mode")

        if mode == "title":
            title_match = re.search(
                r"<title[^>]*>(.*?)</title>",
                html,
                re.IGNORECASE | re.DOTALL,
            )
            title = unescape(title_match.group(1).strip()) if title_match else ""
            return ProviderExtractResult(data={"title": title[:max_chars]})

        stripped = re.sub(
            r"<script[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL
        )
        stripped = re.sub(
            r"<style[^>]*>.*?</style>", " ", stripped, flags=re.IGNORECASE | re.DOTALL
        )
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        stripped = unescape(" ".join(stripped.split()))
        return ProviderExtractResult(data={"text": stripped[:max_chars]})

    async def click(
        self,
        *,
        provider_session_id: str | None,
        selector: str,
    ) -> None:
        _ = (provider_session_id, selector)
        raise ValueError("action_not_supported_by_provider")

    async def type(
        self,
        *,
        provider_session_id: str | None,
        selector: str,
        text: str,
        clear_first: bool,
    ) -> None:
        _ = (provider_session_id, selector, text, clear_first)
        raise ValueError("action_not_supported_by_provider")

    async def wait_for(
        self,
        *,
        provider_session_id: str | None,
        selector: str,
        timeout_seconds: float,
    ) -> None:
        _ = (provider_session_id, selector, timeout_seconds)
        raise ValueError("action_not_supported_by_provider")

    async def screenshot(
        self,
        *,
        provider_session_id: str | None,
    ) -> ProviderScreenshotResult:
        _ = provider_session_id
        raise ValueError("action_not_supported_by_provider")
