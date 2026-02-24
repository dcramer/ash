from __future__ import annotations

from types import SimpleNamespace

from typer.testing import CliRunner

from ash.browser.types import BrowserActionResult
from ash.cli.commands import browser as browser_cmd


class _FakeBrowserManager:
    def __init__(self, fail_action: str | None = None) -> None:
        self.fail_action = fail_action

    async def execute_action(  # noqa: PLR0913
        self,
        *,
        action: str,
        effective_user_id: str,
        provider_name: str | None = None,
        session_id: str | None = None,
        session_name: str | None = None,
        params: dict | None = None,
    ) -> BrowserActionResult:
        _ = (effective_user_id, provider_name, session_id, session_name, params)
        if self.fail_action == action:
            return BrowserActionResult(
                ok=False,
                action=action,
                error_code="action_failed",
                error_message=f"{action} failed",
            )

        if action == "session.start":
            return BrowserActionResult(
                ok=True,
                action=action,
                session_id="s-123",
                provider="sandbox",
            )
        if action == "page.goto":
            return BrowserActionResult(
                ok=True,
                action=action,
                session_id="s-123",
                provider="sandbox",
                page_url="https://example.com",
            )
        if action == "page.extract":
            mode = (params or {}).get("mode")
            if mode == "title":
                return BrowserActionResult(
                    ok=True,
                    action=action,
                    session_id="s-123",
                    provider="sandbox",
                    data={"title": "Example Domain"},
                )
            return BrowserActionResult(
                ok=True,
                action=action,
                session_id="s-123",
                provider="sandbox",
                data={"text": "Example body text"},
            )
        if action == "page.screenshot":
            return BrowserActionResult(
                ok=True,
                action=action,
                session_id="s-123",
                provider="sandbox",
                artifact_refs=["/workspace/browser/s-123/screenshot.png"],
            )
        if action == "session.archive":
            return BrowserActionResult(
                ok=True,
                action=action,
                session_id="s-123",
                provider="sandbox",
            )

        return BrowserActionResult(ok=True, action=action)


class TestBrowserSmokeCommand:
    def test_smoke_success(self, monkeypatch):
        monkeypatch.setattr(browser_cmd, "load_config", lambda _path: SimpleNamespace())
        monkeypatch.setattr(
            browser_cmd,
            "create_browser_manager",
            lambda _cfg: _FakeBrowserManager(),
        )

        runner = CliRunner()
        result = runner.invoke(browser_cmd.app, ["smoke", "https://example.com"])

        assert result.exit_code == 0
        assert "Browser smoke passed" in result.stdout

    def test_smoke_failure_surfaces_action_error(self, monkeypatch):
        monkeypatch.setattr(browser_cmd, "load_config", lambda _path: SimpleNamespace())
        monkeypatch.setattr(
            browser_cmd,
            "create_browser_manager",
            lambda _cfg: _FakeBrowserManager(fail_action="page.goto"),
        )

        runner = CliRunner()
        result = runner.invoke(browser_cmd.app, ["smoke", "https://example.com"])

        assert result.exit_code == 1
        assert "page.goto failed" in result.stdout
