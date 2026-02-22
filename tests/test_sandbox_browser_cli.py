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
from ash.config.models import AshConfig, BrowserConfig, ModelConfig
from ash.integrations import BrowserIntegration, IntegrationContext, IntegrationRuntime
from ash.integrations.rpc import active_rpc_server


def _runner(env: dict[str, str]) -> CliRunner:
    return CliRunner(env=env)


def _ok_result(action: str, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {"ok": True, "action": action}
    payload.update(extra)
    return payload


class TestBrowserSandboxCli:
    def test_start_session(self) -> None:
        runner = _runner({"ASH_USER_ID": "u1"})
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
        runner = _runner({"ASH_USER_ID": "u1"})
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
        runner = _runner({"ASH_USER_ID": "u1"})
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
            enabled=True, provider="sandbox", state_dir=browser_state
        ),
    )
    browser_manager = create_browser_manager(config)
    components = SimpleNamespace(browser_manager=browser_manager)
    context = IntegrationContext(
        config=config,
        components=components,  # type: ignore[arg-type]
        mode="chat",
    )
    runtime = IntegrationRuntime([BrowserIntegration()])
    await runtime.setup(context)

    runner = _runner({"ASH_RPC_SOCKET": str(socket_path), "ASH_USER_ID": "user-1"})

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
