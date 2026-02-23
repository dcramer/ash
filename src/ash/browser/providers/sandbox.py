"""Sandbox-backed browser provider.

This provider never runs browser automation in the host process.
All actions execute inside the shared sandbox container via SandboxExecutor.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets
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
from ash.sandbox.executor import ExecutionResult, SandboxExecutor

logger = logging.getLogger("browser")


@dataclass(slots=True)
class _RemoteSandboxRuntime:
    port: int
    pid: int
    base_dir: str


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
        self._sessions: set[str] = set()
        self._runtime: _RemoteSandboxRuntime | None = None
        self._runtime_lock = asyncio.Lock()
        self._runtime_base_dir: str | None = None
        self.runs_in_sandbox_executor = executor is not None

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
                metadata={"engine": "playwright", "browser": "chromium"},
            )
        await self._ensure_runtime()
        self._sessions.add(session_id)
        return ProviderStartResult(
            provider_session_id=session_id,
            metadata={"engine": "playwright", "browser": "chromium"},
        )

    async def close_session(self, *, provider_session_id: str | None) -> None:
        if not provider_session_id:
            return None
        self._sessions.discard(provider_session_id)
        await self._shutdown_runtime_if_idle()
        return None

    async def warmup(self) -> None:
        """Boot and verify warm browser runtime ahead of first user action."""
        await self._ensure_runtime()

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
        port = await self._resolve_session_port(provider_session_id)
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
            args=[str(port), url, str(timeout_seconds)],
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
        port = await self._resolve_session_port(provider_session_id)
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
                str(port),
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
        port = await self._resolve_session_port(provider_session_id)
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
            args=[str(port)],
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
        port = await self._resolve_session_port(provider_session_id)
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
                str(port),
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

    async def _resolve_session_port(self, provider_session_id: str | None) -> int:
        if not provider_session_id:
            raise ValueError("session_not_found")
        if provider_session_id not in self._sessions:
            # Session state is persisted by manager and provider state is process-local.
            # Rehydrate lazily so active sessions continue across service restarts.
            self._sessions.add(provider_session_id)
            logger.info(
                "browser_sandbox_session_rehydrated",
                extra={
                    "browser.provider": "sandbox",
                    "browser.session_id": provider_session_id,
                },
            )
        runtime = await self._ensure_runtime()
        return runtime.port

    async def _run_json(
        self,
        *,
        code: str,
        args: list[str],
        deadline_seconds: int,
    ) -> dict[str, Any]:
        arg_text = " ".join(shlex.quote(arg) for arg in args)
        command = f"python -c {shlex.quote(code)} {arg_text}".strip()
        result = await self._execute_sandbox_command(
            command,
            phase="python_action",
            timeout_seconds=deadline_seconds,
            reuse_container=True,
            environment={"NODE_OPTIONS": "--no-warnings"},
        )
        if not result.success:
            stderr = result.stderr.strip()
            if stderr:
                cleaned_lines = [
                    line
                    for line in stderr.splitlines()
                    if not line.startswith("(node:")
                    and not line.startswith("(Use `node --trace-deprecation")
                ]
                stderr = "\n".join(cleaned_lines).strip()
            message = stderr or result.stdout.strip() or "unknown error"
            raise ValueError(f"sandbox_browser_action_failed: {message}")
        output = (result.stdout or "").strip()
        if not output:
            return {}
        try:
            return self._parse_json_output(output)
        except Exception as e:
            raise ValueError(f"sandbox_browser_parse_failed: {output[:200]}") from e

    def _parse_json_output(self, output: str) -> dict[str, Any]:
        """Parse JSON output even when non-JSON noise is present."""
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        for line in reversed(output.splitlines()):
            candidate = line.strip()
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("no_json_payload_found")

    def _pick_port(self) -> int:
        used_ports: set[int] = set()
        if self._runtime is not None:
            used_ports.add(self._runtime.port)
        for _ in range(16):
            candidate = 20000 + secrets.randbelow(20000)
            if candidate not in used_ports:
                return candidate
        return 20000 + secrets.randbelow(20000)

    async def _ensure_runtime(self) -> _RemoteSandboxRuntime:
        async with self._runtime_lock:
            if self._runtime and await self._is_runtime_healthy(self._runtime):
                return self._runtime
            if self._runtime is not None:
                logger.warning(
                    "browser_sandbox_runtime_unhealthy",
                    extra={
                        "browser.provider": "sandbox",
                        "browser.runtime_port": self._runtime.port,
                        "browser.runtime_pid": self._runtime.pid,
                    },
                )
                await self._kill_runtime(self._runtime)
                self._runtime = None
            return await self._launch_runtime()

    async def _launch_runtime(self) -> _RemoteSandboxRuntime:
        base_dir = await self._resolve_runtime_base_dir()
        last_error = "sandbox_browser_launch_failed: unknown startup failure"
        launch_deadline = (
            asyncio.get_running_loop().time() + self._MAX_LAUNCH_TOTAL_SECONDS
        )
        logger.info(
            "browser_sandbox_runtime_starting",
            extra={
                "browser.provider": "sandbox",
            },
        )
        for attempt in range(1, self._MAX_LAUNCH_ATTEMPTS + 1):
            if asyncio.get_running_loop().time() >= launch_deadline:
                last_error = (
                    "sandbox_browser_launch_failed: startup_deadline_exceeded:"
                    f"{int(self._MAX_LAUNCH_TOTAL_SECONDS)}s"
                )
                break
            port = self._pick_port()
            launch_cmd = (
                f"mkdir -p {shlex.quote(base_dir)} && "
                "nohup chromium "
                "--headless=new "
                "--no-sandbox "
                "--disable-dev-shm-usage "
                "--disable-gpu "
                "--remote-debugging-address=127.0.0.1 "
                f"--remote-debugging-port={port} "
                f"--user-data-dir={shlex.quote(base_dir)}/profile "
                "about:blank "
                f"> {shlex.quote(base_dir)}/chromium.log 2>&1 & echo $!"
            )
            result = await self._execute_sandbox_command(
                command=f"bash -lc {shlex.quote(launch_cmd)}",
                phase=f"runtime_launch_attempt_{attempt}",
                timeout_seconds=20,
                reuse_container=True,
            )
            if not result.success:
                last_error = (
                    "sandbox_browser_launch_failed: "
                    f"{result.stderr.strip() or result.stdout.strip() or 'failed to start chromium in sandbox'}"
                )
                continue
            lines = (result.stdout or "").strip().splitlines()
            pid_text = lines[-1].strip() if lines else ""
            if not pid_text.isdigit():
                last_error = (
                    f"sandbox_browser_launch_failed: invalid chromium pid '{pid_text}'"
                )
                continue
            runtime = _RemoteSandboxRuntime(
                port=port,
                pid=int(pid_text),
                base_dir=base_dir,
            )
            try:
                await self._wait_for_cdp_ready(runtime=runtime)
            except ValueError as e:
                last_error = str(e)
                await self._kill_runtime(runtime)
                continue
            self._runtime = runtime
            logger.info(
                "browser_sandbox_runtime_ready",
                extra={
                    "browser.provider": "sandbox",
                    "browser.runtime_port": runtime.port,
                    "browser.runtime_pid": runtime.pid,
                },
            )
            return runtime
        raise ValueError(last_error)

    async def _resolve_runtime_base_dir(self) -> str:
        """Pick a writable runtime directory for Chromium profile/log data."""
        if self._runtime_base_dir:
            return self._runtime_base_dir
        candidates = (
            "/home/sandbox/.cache/ash-browser/runtime",
            "/tmp/ash-browser/runtime",  # noqa: S108 - sandbox-local ephemeral runtime
            "/workspace/.ash-browser/runtime",
        )
        for base_dir in candidates:
            probe = await self._execute_sandbox_command(
                command=(
                    "bash -lc "
                    + shlex.quote(
                        f"mkdir -p {shlex.quote(base_dir)} && test -w {shlex.quote(base_dir)}"
                    )
                ),
                phase="runtime_dir_probe",
                timeout_seconds=8,
                reuse_container=True,
            )
            if probe.success:
                self._runtime_base_dir = base_dir
                return base_dir
        raise ValueError("sandbox_browser_launch_failed: no_writable_runtime_dir")

    async def _shutdown_runtime_if_idle(self) -> None:
        """Stop warm runtime when no sessions remain to avoid orphan processes."""
        if self._sessions:
            return
        async with self._runtime_lock:
            if self._sessions or self._runtime is None:
                return
            runtime = self._runtime
            self._runtime = None
            await self._kill_runtime(runtime)
            logger.info(
                "browser_sandbox_runtime_stopped",
                extra={
                    "browser.provider": "sandbox",
                    "browser.runtime_port": runtime.port,
                    "browser.runtime_pid": runtime.pid,
                },
            )

    async def _is_runtime_healthy(self, runtime: _RemoteSandboxRuntime) -> bool:
        http_ok, _ = await self._probe_http_ready(runtime.port, timeout_seconds=2.0)
        if not http_ok:
            return False
        ws_ok, _ = await self._probe_cdp_handshake(runtime.port, timeout_seconds=3.0)
        return ws_ok

    async def _kill_runtime(self, runtime: _RemoteSandboxRuntime) -> None:
        _ = await self._execute_sandbox_command(
            command=f"bash -lc {shlex.quote(f'kill {runtime.pid} >/dev/null 2>&1 || true')}",
            phase="runtime_kill",
            timeout_seconds=5,
            reuse_container=True,
        )

    async def _wait_for_cdp_ready(self, *, runtime: _RemoteSandboxRuntime) -> None:
        http_ok, http_probe = await self._probe_http_ready(
            runtime.port, timeout_seconds=self._HTTP_READY_TIMEOUT_SECONDS
        )
        if not http_ok:
            raise ValueError(
                await self._build_cdp_not_ready_error(
                    runtime=runtime,
                    phase="http",
                    probe_details=http_probe,
                )
            )
        ws_ok, ws_probe = await self._probe_cdp_handshake(
            runtime.port, timeout_seconds=self._WS_READY_TIMEOUT_SECONDS
        )
        if ws_ok:
            return
        raise ValueError(
            await self._build_cdp_not_ready_error(
                runtime=runtime,
                phase="ws",
                probe_details=ws_probe,
            )
        )

    async def _probe_http_ready(
        self, port: int, *, timeout_seconds: float
    ) -> tuple[bool, str]:
        probe_script = """
import sys, time, urllib.request
port = sys.argv[1]
deadline = time.time() + float(sys.argv[2])
url = f"http://127.0.0.1:{port}/json/version"
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=1.0) as resp:
            if resp.status == 200 and b"webSocketDebuggerUrl" in resp.read():
                print("ok")
                raise SystemExit(0)
    except Exception:
        time.sleep(0.2)
raise SystemExit(1)
"""
        probe = await self._execute_sandbox_command(
            command=f"python -c {shlex.quote(probe_script)} {port} {timeout_seconds}",
            phase="http_probe",
            timeout_seconds=max(5, int(timeout_seconds) + 5),
            reuse_container=True,
        )
        details = (probe.stderr or probe.stdout or "").strip()
        return probe.success, details

    async def _probe_cdp_handshake(
        self, port: int, *, timeout_seconds: float
    ) -> tuple[bool, str]:
        probe_script = """
import asyncio, sys
from playwright.async_api import async_playwright
port = sys.argv[1]
timeout_s = float(sys.argv[2])
async def main():
    pw = await async_playwright().start()
    try:
        browser = await pw.chromium.connect_over_cdp(
            f"http://127.0.0.1:{port}",
            timeout=int(max(0.1, timeout_s) * 1000),
        )
        await browser.close()
        print("ok")
    finally:
        await pw.stop()
asyncio.run(main())
"""
        probe = await self._execute_sandbox_command(
            command=f"python -c {shlex.quote(probe_script)} {port} {timeout_seconds}",
            phase="cdp_probe",
            timeout_seconds=max(5, int(timeout_seconds) + 5),
            reuse_container=True,
            environment={"NODE_OPTIONS": "--no-warnings"},
        )
        details = (probe.stderr or probe.stdout or "").strip()
        return probe.success, details

    async def _build_cdp_not_ready_error(
        self,
        *,
        runtime: _RemoteSandboxRuntime,
        phase: str,
        probe_details: str,
    ) -> str:
        proc_alive = await self._execute_sandbox_command(
            command=f"bash -lc {shlex.quote(f'kill -0 {runtime.pid} >/dev/null 2>&1 && echo alive || echo dead')}",
            phase="runtime_alive_probe",
            timeout_seconds=5,
            reuse_container=True,
        )
        log_tail_cmd = f"bash -lc {shlex.quote(f'tail -n 40 {shlex.quote(runtime.base_dir)}/chromium.log 2>/dev/null || true')}"
        log_tail = await self._execute_sandbox_command(
            command=log_tail_cmd,
            phase="runtime_log_tail",
            timeout_seconds=10,
            reuse_container=True,
        )
        details = (log_tail.stdout or log_tail.stderr or "").strip()
        alive_text = (proc_alive.stdout or "").strip()
        prefix = f"sandbox_browser_launch_failed: cdp_not_ready:{phase}"
        if details:
            return (
                f"{prefix}; process={alive_text or 'unknown'}; chromium_log={details}"
            )
        if probe_details:
            return f"{prefix}; process={alive_text or 'unknown'}; probe={probe_details}"
        return f"{prefix}; process={alive_text or 'unknown'}; probe=unavailable"

    async def _execute_sandbox_command(
        self,
        command: str,
        *,
        phase: str,
        timeout_seconds: int,
        reuse_container: bool,
        environment: dict[str, str] | None = None,
    ) -> ExecutionResult:
        executor = self._require_executor()
        # Guard against hangs in container/image startup paths that can occur
        # before sandbox command-level timeouts are applied.
        outer_timeout = max(5, timeout_seconds + 10)
        try:
            return await asyncio.wait_for(
                executor.execute(
                    command,
                    timeout=timeout_seconds,
                    reuse_container=reuse_container,
                    environment=environment,
                ),
                timeout=outer_timeout,
            )
        except TimeoutError as e:
            raise ValueError(
                f"sandbox_browser_action_timeout:{phase}:{outer_timeout}s"
            ) from e

    _MAX_LAUNCH_ATTEMPTS = 3
    _MAX_LAUNCH_TOTAL_SECONDS = 45.0
    _HTTP_READY_TIMEOUT_SECONDS = 12.0
    _WS_READY_TIMEOUT_SECONDS = 8.0
