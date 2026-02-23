"""Sandbox-backed browser provider.

This provider never runs browser automation in the host process.
All actions execute inside the shared sandbox container via SandboxExecutor.
"""

from __future__ import annotations

import base64
import json
import shlex
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from ash.browser.providers.base import (
    ProviderExtractResult,
    ProviderGotoResult,
    ProviderScreenshotResult,
    ProviderStartResult,
)
from ash.sandbox.executor import SandboxExecutor


@dataclass(slots=True)
class _RemoteSandboxSession:
    port: int
    pid: int


class SandboxBrowserProvider:
    """Browser provider implemented via sandbox container execution."""

    name = "sandbox"

    def __init__(
        self,
        *,
        headless: bool = True,
        browser_channel: str = "chromium",
        viewport_width: int = 1280,
        viewport_height: int = 720,
        executor: SandboxExecutor | None = None,
    ) -> None:
        _ = (headless, browser_channel, viewport_width, viewport_height)
        self._executor = executor
        self._sessions: dict[str, _RemoteSandboxSession] = {}
        self.runs_in_sandbox_executor = executor is not None

    async def start_session(
        self,
        *,
        session_id: str,
        profile_name: str | None,
    ) -> ProviderStartResult:
        _ = profile_name
        executor = self._require_executor()
        if session_id in self._sessions:
            return ProviderStartResult(
                provider_session_id=session_id,
                metadata={"engine": "playwright", "browser": "chromium"},
            )

        port = 21000 + (abs(hash(session_id)) % 10000)
        base_dir = f"/home/sandbox/.cache/ash-browser/{session_id}"
        launch_cmd = (
            f"mkdir -p {shlex.quote(base_dir)} && "
            "nohup chromium "
            "--headless=new "
            "--disable-gpu "
            "--remote-debugging-address=127.0.0.1 "
            f"--remote-debugging-port={port} "
            f"--user-data-dir={shlex.quote(base_dir)}/profile "
            "about:blank "
            f"> {shlex.quote(base_dir)}/chromium.log 2>&1 & echo $!"
        )
        result = await executor.execute(
            f"bash -lc {shlex.quote(launch_cmd)}",
            timeout=20,
            reuse_container=True,
        )
        if not result.success:
            raise ValueError(
                "sandbox_browser_launch_failed: failed to start chromium in sandbox"
            )
        lines = (result.stdout or "").strip().splitlines()
        pid_text = lines[-1].strip() if lines else ""
        if not pid_text.isdigit():
            raise ValueError(
                f"sandbox_browser_launch_failed: invalid chromium pid '{pid_text}'"
            )
        self._sessions[session_id] = _RemoteSandboxSession(port=port, pid=int(pid_text))
        return ProviderStartResult(
            provider_session_id=session_id,
            metadata={"engine": "playwright", "browser": "chromium"},
        )

    async def close_session(self, *, provider_session_id: str | None) -> None:
        if not provider_session_id:
            return None
        session = self._sessions.pop(provider_session_id, None)
        if session is None:
            return None
        executor = self._require_executor()
        _ = await executor.execute(
            f"bash -lc {shlex.quote(f'kill {session.pid} >/dev/null 2>&1 || true')}",
            timeout=10,
            reuse_container=True,
        )
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
        payload = await self._run_json(
            code="""
import asyncio, json, sys
from playwright.async_api import async_playwright
port, url, timeout_s = sys.argv[1], sys.argv[2], float(sys.argv[3])
async def main():
    pw = await async_playwright().start()
    browser = await pw.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
    try:
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=int(max(1.0, timeout_s) * 1000))
        title = await page.title()
        html = await page.content()
        print(json.dumps({"url": page.url, "title": title, "html": html}, ensure_ascii=True))
    finally:
        await browser.close()
        await pw.stop()
asyncio.run(main())
""",
            args=[str(session.port), url, str(timeout_seconds)],
            deadline_seconds=max(30, int(timeout_seconds) + 5),
        )
        return ProviderGotoResult(
            url=str(payload.get("url") or url),
            title=str(payload.get("title") or "") or None,
            html=str(payload.get("html") or "") or None,
        )

    async def extract(
        self,
        *,
        provider_session_id: str | None,
        html: str | None,
        mode: str,
        selector: str | None,
        max_chars: int,
    ) -> ProviderExtractResult:
        _ = html
        session = self._require_session(provider_session_id)
        if mode not in {"text", "title"}:
            raise ValueError("unsupported_extract_mode")
        payload = await self._run_json(
            code="""
import asyncio, json, sys
from playwright.async_api import async_playwright
port, mode, selector, max_chars = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
selector = None if selector == "__NONE__" else selector
async def main():
    pw = await async_playwright().start()
    browser = await pw.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
    try:
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        if mode == "title":
            title = await page.title()
            print(json.dumps({"title": (title or "")[:max_chars]}, ensure_ascii=True))
            return
        if selector:
            text = await page.locator(selector).first.inner_text(timeout=3000)
        else:
            text = await page.evaluate("() => document.body ? document.body.innerText || '' : ''")
        print(json.dumps({"text": (text or "")[:max_chars]}, ensure_ascii=True))
    finally:
        await browser.close()
        await pw.stop()
asyncio.run(main())
""",
            args=[
                str(session.port),
                mode,
                selector or "__NONE__",
                str(max(1, max_chars)),
            ],
            deadline_seconds=30,
        )
        return ProviderExtractResult(data=payload)

    async def click(
        self,
        *,
        provider_session_id: str | None,
        selector: str,
    ) -> None:
        await self._click_type_wait(
            action="click",
            provider_session_id=provider_session_id,
            selector=selector,
        )

    async def type(
        self,
        *,
        provider_session_id: str | None,
        selector: str,
        text: str,
        clear_first: bool,
    ) -> None:
        await self._click_type_wait(
            action="type",
            provider_session_id=provider_session_id,
            selector=selector,
            text=text,
            clear_first=clear_first,
        )

    async def wait_for(
        self,
        *,
        provider_session_id: str | None,
        selector: str,
        timeout_seconds: float,
    ) -> None:
        await self._click_type_wait(
            action="wait_for",
            provider_session_id=provider_session_id,
            selector=selector,
            timeout_seconds=timeout_seconds,
        )

    async def screenshot(
        self,
        *,
        provider_session_id: str | None,
    ) -> ProviderScreenshotResult:
        session = self._require_session(provider_session_id)
        payload = await self._run_json(
            code="""
import asyncio, base64, json, sys
from playwright.async_api import async_playwright
port = sys.argv[1]
async def main():
    pw = await async_playwright().start()
    browser = await pw.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
    try:
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        image = await page.screenshot(type="png", full_page=True)
        print(json.dumps({"image_b64": base64.b64encode(image).decode("ascii")}, ensure_ascii=True))
    finally:
        await browser.close()
        await pw.stop()
asyncio.run(main())
""",
            args=[str(session.port)],
            deadline_seconds=30,
        )
        image_b64 = str(payload.get("image_b64") or "")
        if not image_b64:
            raise ValueError("screenshot_failed")
        return ProviderScreenshotResult(
            image_bytes=base64.b64decode(image_b64),
            mime_type="image/png",
        )

    async def _click_type_wait(
        self,
        *,
        action: str,
        provider_session_id: str | None,
        selector: str,
        text: str = "",
        clear_first: bool = True,
        timeout_seconds: float = 30.0,
    ) -> None:
        session = self._require_session(provider_session_id)
        _ = await self._run_json(
            code="""
import asyncio, sys
from playwright.async_api import async_playwright
port, action, selector, text, clear_first, timeout_s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] == "1", float(sys.argv[6])
async def main():
    pw = await async_playwright().start()
    browser = await pw.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
    try:
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        if action == "click":
            await page.locator(selector).first.click(timeout=10000)
        elif action == "type":
            locator = page.locator(selector).first
            if clear_first:
                await locator.fill("")
            await locator.type(text, delay=0)
        else:
            await page.wait_for_selector(selector, state="visible", timeout=int(max(0.1, timeout_s) * 1000))
        print("{}")
    finally:
        await browser.close()
        await pw.stop()
asyncio.run(main())
""",
            args=[
                str(session.port),
                action,
                selector,
                text,
                "1" if clear_first else "0",
                str(timeout_seconds),
            ],
            deadline_seconds=max(15, int(timeout_seconds) + 5),
        )

    def _require_executor(self) -> SandboxExecutor:
        if self._executor is None:
            raise ValueError(
                "sandbox_executor_required: browser provider 'sandbox' requires "
                "container-backed executor wiring"
            )
        return self._executor

    def _require_session(
        self, provider_session_id: str | None
    ) -> _RemoteSandboxSession:
        if not provider_session_id:
            raise ValueError("session_not_found")
        session = self._sessions.get(provider_session_id)
        if session is None:
            raise ValueError("session_not_found")
        return session

    async def _run_json(
        self,
        *,
        code: str,
        args: list[str],
        deadline_seconds: int,
    ) -> dict[str, Any]:
        executor = self._require_executor()
        arg_text = " ".join(shlex.quote(arg) for arg in args)
        command = f"python -c {shlex.quote(code)} {arg_text}".strip()
        result = await executor.execute(
            command,
            timeout=deadline_seconds,
            reuse_container=True,
        )
        if not result.success:
            message = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise ValueError(f"sandbox_browser_action_failed: {message}")
        output = (result.stdout or "").strip()
        if not output:
            return {}
        try:
            return json.loads(output)
        except Exception as e:
            raise ValueError(f"sandbox_browser_parse_failed: {output[:200]}") from e
