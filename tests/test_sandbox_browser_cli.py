"""Tests for sandboxed CLI browser commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from ash_sandbox_cli.commands.browser import app
from typer.testing import CliRunner

from ash.browser import create_browser_manager
from ash.config.models import (
    AshConfig,
    BrowserConfig,
    BrowserSandboxConfig,
    ModelConfig,
)
from ash.context_token import get_default_context_token_service
from ash.integrations import BrowserIntegration, IntegrationContext, IntegrationRuntime
from ash.integrations.rpc import active_rpc_server
from ash.sandbox.executor import ExecutionResult


def _runner(env: dict[str, str]) -> CliRunner:
    return CliRunner(env=env)


def _context_token(
    *,
    effective_user_id: str = "u1",
    provider: str | None = None,
    chat_id: str | None = None,
) -> str:
    return get_default_context_token_service().issue(
        effective_user_id=effective_user_id,
        provider=provider,
        chat_id=chat_id,
    )


def _ok_result(action: str, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {"ok": True, "action": action}
    payload.update(extra)
    return payload


class TestBrowserSandboxCli:
    def test_start_session(self) -> None:
        runner = _runner({"ASH_CONTEXT_TOKEN": _context_token(effective_user_id="u1")})
        with patch("ash_sandbox_cli.commands.browser.rpc_call") as mock_rpc:
            mock_rpc.return_value = _ok_result(
                "session.start", session_id="s1", data={"status": "active"}
            )
            result = runner.invoke(app, ["start", "--name", "work"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        mock_rpc.assert_called_once_with(
            "browser.session.start",
            {"user_id": "u1", "session_name": "work"},
        )

    def test_extract_non_ok_exits_nonzero(self) -> None:
        runner = _runner({"ASH_CONTEXT_TOKEN": _context_token(effective_user_id="u1")})
        with patch("ash_sandbox_cli.commands.browser.rpc_call") as mock_rpc:
            mock_rpc.return_value = {
                "ok": False,
                "action": "page.extract",
                "error_code": "session_not_found",
            }
            result = runner.invoke(app, ["extract", "--session-id", "missing"])

        assert result.exit_code == 1
        payload = json.loads(result.stdout)
        assert payload["ok"] is False
        assert payload["error_code"] == "session_not_found"

    def test_goto_includes_url_and_provider(self) -> None:
        runner = _runner({"ASH_CONTEXT_TOKEN": _context_token(effective_user_id="u1")})
        with patch("ash_sandbox_cli.commands.browser.rpc_call") as mock_rpc:
            mock_rpc.return_value = _ok_result(
                "page.goto", page_url="https://example.com"
            )
            result = runner.invoke(
                app,
                [
                    "goto",
                    "https://example.com",
                    "--provider",
                    "sandbox",
                    "--session-id",
                    "s1",
                ],
            )

        assert result.exit_code == 0
        mock_rpc.assert_called_once_with(
            "browser.page.goto",
            {
                "user_id": "u1",
                "provider": "sandbox",
                "session_id": "s1",
                "url": "https://example.com",
            },
        )


@pytest.mark.asyncio
async def test_browser_cli_end_to_end_via_real_rpc(tmp_path: Path) -> None:
    """Start/list/extract/archive via real socket RPC + browser integration."""
    socket_path = tmp_path / "rpc.sock"
    browser_state = tmp_path / "browser-state"

    config = AshConfig(
        models={"default": ModelConfig(provider="openai", model="gpt-5-mini")},
        browser=BrowserConfig(
            enabled=True,
            provider="sandbox",
            state_dir=browser_state,
            sandbox=BrowserSandboxConfig(
                runtime_required=False,
            ),
        ),
    )
    browser_manager = create_browser_manager(config)
    provider = browser_manager._providers["sandbox"]

    async def _fake_host_command(
        args: list[str], *, timeout_seconds: int
    ) -> ExecutionResult:
        _ = timeout_seconds
        if args[:3] == ["docker", "image", "inspect"]:
            return ExecutionResult(exit_code=0, stdout="[]", stderr="")
        if args[:4] == ["docker", "inspect", "-f", "{{.State.Running}}"]:
            return ExecutionResult(exit_code=1, stdout="", stderr="missing")
        if args[:2] == ["docker", "create"]:
            return ExecutionResult(exit_code=0, stdout="created", stderr="")
        if args[:2] == ["docker", "start"]:
            return ExecutionResult(exit_code=0, stdout="started", stderr="")
        if args[:2] == ["docker", "port"]:
            return ExecutionResult(exit_code=0, stdout="127.0.0.1:39422\n", stderr="")
        if args[:3] == ["docker", "rm", "-f"]:
            return ExecutionResult(exit_code=0, stdout="", stderr="")
        if args[:2] == ["docker", "exec"]:
            return ExecutionResult(exit_code=0, stdout="{}\n", stderr="")
        return ExecutionResult(exit_code=0, stdout="{}\n", stderr="")

    async def _noop_wait(*, runtime: object) -> None:
        _ = runtime

    async def _always_healthy(runtime: object) -> bool:
        _ = runtime
        return True

    async def _fake_run_json(
        *,
        code: str,
        args: list[str],
        deadline_seconds: int,
        provider_session_id: str | None = None,
        runtime: object | None = None,
    ) -> dict[str, object]:
        _ = (args, deadline_seconds, provider_session_id, runtime)
        if "/json/list" in code:
            return {"exists": True}
        if "target_id_missing" in code:
            return {"target_id": "target-ops"}
        if "mode, selector, max_chars" in code:
            return {"text": ""}
        return {}

    provider._execute_host_command = _fake_host_command  # type: ignore[assignment]
    provider._wait_for_cdp_ready = _noop_wait  # type: ignore[assignment]
    provider._run_json = _fake_run_json  # type: ignore[assignment]
    provider._is_runtime_healthy = _always_healthy  # type: ignore[assignment]
    components = SimpleNamespace(browser_manager=browser_manager)
    context = IntegrationContext(
        config=config,
        components=components,  # type: ignore[arg-type]
        mode="chat",
    )
    runtime = IntegrationRuntime([BrowserIntegration()])
    await runtime.setup(context)

    runner = _runner(
        {
            "ASH_RPC_SOCKET": str(socket_path),
            "ASH_CONTEXT_TOKEN": _context_token(effective_user_id="user-1"),
        }
    )

    async with active_rpc_server(
        runtime=runtime, context=context, socket_path=socket_path
    ):
        start = await asyncio.to_thread(runner.invoke, app, ["start", "--name", "ops"])
        assert start.exit_code == 0
        start_payload = json.loads(start.stdout)
        session_id = str(start_payload["session_id"])

        listed = await asyncio.to_thread(runner.invoke, app, ["list"])
        assert listed.exit_code == 0
        list_payload = json.loads(listed.stdout)
        assert list_payload["ok"] is True
        assert list_payload["data"]["count"] == 1

        extracted = await asyncio.to_thread(
            runner.invoke,
            app,
            ["extract", "--session-id", session_id, "--mode", "text"],
        )
        assert extracted.exit_code == 0
        extract_payload = json.loads(extracted.stdout)
        assert extract_payload["ok"] is True
        assert extract_payload["data"]["text"] == ""

        archived = await asyncio.to_thread(
            runner.invoke,
            app,
            ["archive", "--session-id", session_id],
        )
        assert archived.exit_code == 0
        archive_payload = json.loads(archived.stdout)
        assert archive_payload["ok"] is True
