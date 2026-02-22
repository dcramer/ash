"""Sandbox-backed browser provider using Chromium via Playwright."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from html import unescape
from typing import Any
from urllib.parse import urlparse

from ash.browser.providers.base import (
    ProviderExtractResult,
    ProviderGotoResult,
    ProviderScreenshotResult,
    ProviderStartResult,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _SandboxSession:
    playwright: Any
    browser: Any
    context: Any
    page: Any


class SandboxBrowserProvider:
    """Sandbox browser provider backed by Playwright Chromium sessions."""

    name = "sandbox"

    def __init__(
        self,
        *,
        headless: bool = True,
        browser_channel: str = "chromium",
        viewport_width: int = 1280,
        viewport_height: int = 720,
    ) -> None:
        self._headless = headless
        self._browser_channel = browser_channel
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._sessions: dict[str, _SandboxSession] = {}

    async def start_session(
        self,
        *,
        session_id: str,
        profile_name: str | None,
    ) -> ProviderStartResult:
        _ = profile_name
        if session_id in self._sessions:
            return ProviderStartResult(
                provider_session_id=session_id,
                metadata={"engine": "playwright", "browser": self._browser_channel},
            )

        session = await self._create_session()
        self._sessions[session_id] = session
        return ProviderStartResult(
            provider_session_id=session_id,
            metadata={"engine": "playwright", "browser": self._browser_channel},
        )

    async def close_session(self, *, provider_session_id: str | None) -> None:
        if not provider_session_id:
            return None
        session = self._sessions.pop(provider_session_id, None)
        if session is None:
            return None
        await self._safe_close(session)
        return None

    async def goto(
        self,
        *,
        provider_session_id: str | None,
        url: str,
        timeout_seconds: float,
    ) -> ProviderGotoResult:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("invalid_url_scheme")
        session = self._require_session(provider_session_id)
        page = session.page
        timeout_ms = int(max(1.0, timeout_seconds) * 1000)
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        title = await page.title()
        html = await page.content()
        return ProviderGotoResult(url=str(page.url), title=title or None, html=html)

    async def extract(
        self,
        *,
        provider_session_id: str | None,
        html: str | None,
        mode: str,
        selector: str | None,
        max_chars: int,
    ) -> ProviderExtractResult:
        if mode not in {"text", "title"}:
            raise ValueError("unsupported_extract_mode")

        session = self._require_session(provider_session_id)
        page = session.page
        limit = max(1, max_chars)

        if mode == "title":
            title = await page.title()
            return ProviderExtractResult(data={"title": (title or "")[:limit]})

        if selector:
            locator = page.locator(selector).first
            try:
                text = await locator.inner_text(timeout=3_000)
            except Exception as e:
                raise ValueError("selector_not_found") from e
            return ProviderExtractResult(data={"text": text[:limit]})

        # Prefer DOM text from live page; fallback to HTML stripping if unavailable.
        try:
            text = await page.evaluate(
                "() => document.body ? document.body.innerText || '' : ''"
            )
            if isinstance(text, str):
                return ProviderExtractResult(data={"text": text[:limit]})
        except Exception:
            logger.debug("browser_extract_dom_text_failed", exc_info=True)

        if not html:
            return ProviderExtractResult(data={"text": ""})
        stripped = re.sub(
            r"<script[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL
        )
        stripped = re.sub(
            r"<style[^>]*>.*?</style>", " ", stripped, flags=re.IGNORECASE | re.DOTALL
        )
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        stripped = unescape(" ".join(stripped.split()))
        return ProviderExtractResult(data={"text": stripped[:limit]})

    async def click(
        self,
        *,
        provider_session_id: str | None,
        selector: str,
    ) -> None:
        session = self._require_session(provider_session_id)
        await session.page.locator(selector).first.click(timeout=10_000)

    async def type(
        self,
        *,
        provider_session_id: str | None,
        selector: str,
        text: str,
        clear_first: bool,
    ) -> None:
        session = self._require_session(provider_session_id)
        locator = session.page.locator(selector).first
        if clear_first:
            await locator.fill("")
        await locator.type(text, delay=0)

    async def wait_for(
        self,
        *,
        provider_session_id: str | None,
        selector: str,
        timeout_seconds: float,
    ) -> None:
        session = self._require_session(provider_session_id)
        timeout_ms = int(max(0.1, timeout_seconds) * 1000)
        await session.page.wait_for_selector(
            selector, state="visible", timeout=timeout_ms
        )

    async def screenshot(
        self,
        *,
        provider_session_id: str | None,
    ) -> ProviderScreenshotResult:
        session = self._require_session(provider_session_id)
        image = await session.page.screenshot(type="png", full_page=True)
        return ProviderScreenshotResult(image_bytes=image, mime_type="image/png")

    def _require_session(self, provider_session_id: str | None) -> _SandboxSession:
        if not provider_session_id:
            raise ValueError("session_not_found")
        session = self._sessions.get(provider_session_id)
        if session is None:
            raise ValueError("session_not_found")
        return session

    async def _create_session(self) -> _SandboxSession:
        try:
            from playwright.async_api import async_playwright
        except Exception as e:
            raise ValueError(
                "playwright_not_installed: run `uv sync --all-groups` and "
                "`uv run playwright install chromium`, then restart ash"
            ) from e

        playwright = await async_playwright().start()
        try:
            browser = await playwright.chromium.launch(
                headless=self._headless,
                channel=self._browser_channel or None,
            )
        except Exception as e:
            await playwright.stop()
            message = str(e).lower()
            if (
                "executable doesn't exist" in message
                or "browser has been closed" in message
                or "failed to launch" in message
            ):
                raise ValueError(
                    "chromium_not_installed: run `uv run playwright install chromium`, "
                    "then restart ash"
                ) from e
            raise

        context = await browser.new_context(
            viewport={
                "width": self._viewport_width,
                "height": self._viewport_height,
            }
        )
        page = await context.new_page()
        return _SandboxSession(
            playwright=playwright,
            browser=browser,
            context=context,
            page=page,
        )

    async def _safe_close(self, session: _SandboxSession) -> None:
        for closer in (
            getattr(session.page, "close", None),
            getattr(session.context, "close", None),
            getattr(session.browser, "close", None),
            getattr(session.playwright, "stop", None),
        ):
            if closer is None:
                continue
            try:
                await closer()
            except Exception:
                logger.debug("browser_session_close_failed", exc_info=True)
