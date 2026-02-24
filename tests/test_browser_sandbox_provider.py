from __future__ import annotations

from typing import Any

import pytest

from ash.browser.providers.sandbox import SandboxBrowserProvider
from ash.sandbox.executor import ExecutionResult


class _HostDockerStub:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.containers: dict[str, bool] = {}

    async def run(self, args: list[str], *, timeout_seconds: int) -> ExecutionResult:
        _ = timeout_seconds
        self.calls.append(args)
        if args[:3] == ["docker", "image", "inspect"]:
            return ExecutionResult(exit_code=0, stdout="[]", stderr="")
        if args[:4] == ["docker", "inspect", "-f", "{{.State.Running}}"]:
            name = args[4]
            if name not in self.containers:
                return ExecutionResult(exit_code=1, stdout="", stderr="missing")
            return ExecutionResult(
                exit_code=0,
                stdout=("true" if self.containers[name] else "false"),
                stderr="",
            )
        if args[:2] == ["docker", "create"]:
            name = args[args.index("--name") + 1]
            self.containers[name] = False
            return ExecutionResult(exit_code=0, stdout=name, stderr="")
        if args[:2] == ["docker", "start"]:
            self.containers[args[2]] = True
            return ExecutionResult(exit_code=0, stdout=args[2], stderr="")
        if args[:2] == ["docker", "port"]:
            return ExecutionResult(exit_code=0, stdout="127.0.0.1:39422\n", stderr="")
        if args[:3] == ["docker", "rm", "-f"]:
            self.containers.pop(args[3], None)
            return ExecutionResult(exit_code=0, stdout="", stderr="")
        if args[:2] == ["docker", "exec"]:
            return ExecutionResult(exit_code=0, stdout="{}\n", stderr="")
        return ExecutionResult(exit_code=0, stdout="", stderr="")


def test_parse_json_output_ignores_noise() -> None:
    provider = SandboxBrowserProvider(executor=None)
    payload = provider._parse_json_output(
        '(node:1) warning\nnot json\n{"ok": true, "value": 1}'
    )
    assert payload["ok"] is True
    assert payload["value"] == 1


@pytest.mark.asyncio
async def test_sandbox_provider_scopes_provider_session_id_and_container_name() -> None:
    stub = _HostDockerStub()
    provider = SandboxBrowserProvider(container_name_prefix="ash-browser-")
    provider._execute_host_command = stub.run  # type: ignore[assignment]
    provider._wait_for_cdp_ready = _noop_wait  # type: ignore[assignment]
    provider._run_json = _fake_run_json  # type: ignore[assignment]

    started = await provider.start_session(
        session_id="s1",
        profile_name=None,
        scope_key="user-123",
    )
    assert started.provider_session_id is not None
    assert started.provider_session_id.endswith(":s1")

    container_name = next(
        args[args.index("--name") + 1]
        for args in stub.calls
        if args[:2] == ["docker", "create"]
    )
    assert container_name.startswith("ash-browser-")
    assert len(container_name) <= 63


@pytest.mark.asyncio
async def test_sandbox_provider_full_flow_with_dedicated_runtime() -> None:
    stub = _HostDockerStub()
    provider = SandboxBrowserProvider()
    provider._execute_host_command = stub.run  # type: ignore[assignment]
    provider._wait_for_cdp_ready = _noop_wait  # type: ignore[assignment]
    provider._run_json = _fake_run_json  # type: ignore[assignment]

    started = await provider.start_session(
        session_id="s1",
        profile_name=None,
        scope_key="user-1",
    )
    assert started.provider_session_id is not None
    session_id = started.provider_session_id

    goto = await provider.goto(
        provider_session_id=session_id,
        url="https://example.com",
        timeout_seconds=2.0,
    )
    assert goto.url == "https://example.com"
    assert goto.title == "Example"

    extracted = await provider.extract(
        provider_session_id=session_id,
        html=None,
        mode="title",
        selector=None,
        max_chars=100,
    )
    assert extracted.data["title"] == "Example"

    shot = await provider.screenshot(provider_session_id=session_id)
    assert shot.mime_type == "image/png"
    assert shot.image_bytes == b"hello"

    await provider.close_session(provider_session_id=session_id)
    assert any(args[:3] == ["docker", "rm", "-f"] for args in stub.calls)


@pytest.mark.asyncio
async def test_sandbox_provider_rehydrates_session_after_restart_like_state() -> None:
    stub = _HostDockerStub()
    provider = SandboxBrowserProvider()
    provider._execute_host_command = stub.run  # type: ignore[assignment]
    provider._wait_for_cdp_ready = _noop_wait  # type: ignore[assignment]
    provider._run_json = _fake_run_json  # type: ignore[assignment]

    started = await provider.start_session(
        session_id="s1",
        profile_name=None,
        scope_key="user-1",
    )
    session_id = started.provider_session_id
    assert session_id is not None
    provider._sessions.clear()

    extracted = await provider.extract(
        provider_session_id=session_id,
        html=None,
        mode="text",
        selector=None,
        max_chars=100,
    )
    assert extracted.data["text"] == "Hello"


@pytest.mark.asyncio
async def test_sandbox_provider_times_out_hung_docker_exec() -> None:
    async def _host_timeout(
        args: list[str], *, timeout_seconds: int
    ) -> ExecutionResult:
        _ = (args, timeout_seconds)
        return ExecutionResult(exit_code=-1, stdout="", stderr="", timed_out=True)

    provider = SandboxBrowserProvider()
    provider._execute_host_command = _host_timeout  # type: ignore[assignment]
    provider._runtime = type("R", (), {"container_name": "ash-browser-timeout"})()
    with pytest.raises(ValueError, match="sandbox_browser_action_timeout:test"):
        await provider._execute_sandbox_command(
            "echo hi",
            phase="test",
            timeout_seconds=1,
            reuse_container=True,
            runtime=provider._runtime,
        )


async def _noop_wait(*, runtime: Any) -> None:
    _ = runtime


async def _fake_run_json(
    *,
    code: str,
    args: list[str],
    deadline_seconds: int,
    provider_session_id: str | None = None,
    runtime: Any | None = None,
) -> dict[str, Any]:
    _ = (args, deadline_seconds, provider_session_id, runtime)
    if "target_id_missing" in code:
        return {"target_id": "target-s1"}
    if "page.goto(" in code:
        return {
            "url": "https://example.com",
            "title": "Example",
            "html": "<html></html>",
        }
    if "mode, selector, max_chars" in code:
        mode = args[2] if len(args) > 2 else "text"
        return {"title": "Example"} if mode == "title" else {"text": "Hello"}
    if "page.screenshot" in code:
        return {"image_b64": "aGVsbG8="}
    if "/json/list" in code:
        return {"exists": True}
    return {}
