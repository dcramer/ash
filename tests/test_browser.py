from __future__ import annotations

from pathlib import Path

import pytest

from ash.browser.manager import BrowserManager
from ash.browser.providers.base import (
    ProviderExtractResult,
    ProviderGotoResult,
    ProviderScreenshotResult,
    ProviderStartResult,
)
from ash.browser.store import BrowserStore
from ash.config.models import AshConfig, BrowserConfig, ModelConfig
from ash.tools.base import ToolContext
from ash.tools.builtin.browser import BrowserTool


class _FakeProvider:
    name = "sandbox"

    async def start_session(self, *, session_id: str, profile_name: str | None):
        _ = profile_name
        return ProviderStartResult(provider_session_id=f"p-{session_id[:8]}")

    async def close_session(self, *, provider_session_id: str | None) -> None:
        _ = provider_session_id

    async def goto(
        self, *, provider_session_id: str | None, url: str, timeout_seconds: float
    ):
        _ = (provider_session_id, timeout_seconds)
        return ProviderGotoResult(
            url=url,
            title="Example",
            html="<html><title>Example</title><body>Hello world</body></html>",
        )

    async def extract(
        self,
        *,
        provider_session_id: str | None,
        html: str | None,
        mode: str,
        selector: str | None,
        max_chars: int,
    ):
        _ = (provider_session_id, selector)
        if mode == "title":
            return ProviderExtractResult(data={"title": "Example"})
        return ProviderExtractResult(data={"text": (html or "")[:max_chars]})

    async def click(self, *, provider_session_id: str | None, selector: str) -> None:
        _ = (provider_session_id, selector)

    async def type(
        self,
        *,
        provider_session_id: str | None,
        selector: str,
        text: str,
        clear_first: bool,
    ) -> None:
        _ = (provider_session_id, selector, text, clear_first)

    async def wait_for(
        self,
        *,
        provider_session_id: str | None,
        selector: str,
        timeout_seconds: float,
    ) -> None:
        _ = (provider_session_id, selector, timeout_seconds)

    async def screenshot(self, *, provider_session_id: str | None):
        _ = provider_session_id
        return ProviderScreenshotResult(image_bytes=b"abc")


class _FakeKernelProvider(_FakeProvider):
    name = "kernel"


def _config() -> AshConfig:
    return AshConfig(
        workspace=Path("tmp-workspace"),
        models={"default": ModelConfig(provider="openai", model="gpt-5.2")},
        browser=BrowserConfig(),
    )


@pytest.mark.asyncio
async def test_browser_manager_session_lifecycle(tmp_path) -> None:
    store = BrowserStore(tmp_path / "browser")
    manager = BrowserManager(
        config=_config(), store=store, providers={"sandbox": _FakeProvider()}
    )

    started = await manager.execute_action(
        action="session.start",
        effective_user_id="u1",
        session_name="work",
        provider_name="sandbox",
    )
    assert started.ok is True
    assert started.session_id is not None

    listed = await manager.execute_action(
        action="session.list",
        effective_user_id="u1",
        provider_name="sandbox",
    )
    assert listed.ok is True
    assert listed.data["count"] == 1

    archived = await manager.execute_action(
        action="session.archive",
        effective_user_id="u1",
        provider_name="sandbox",
        session_id=started.session_id,
    )
    assert archived.ok is True


@pytest.mark.asyncio
async def test_browser_manager_goto_extract_and_screenshot(tmp_path) -> None:
    store = BrowserStore(tmp_path / "browser")
    manager = BrowserManager(
        config=_config(), store=store, providers={"sandbox": _FakeProvider()}
    )

    started = await manager.execute_action(
        action="session.start",
        effective_user_id="u1",
        provider_name="sandbox",
    )
    session_id = started.session_id
    assert session_id

    goto = await manager.execute_action(
        action="page.goto",
        effective_user_id="u1",
        provider_name="sandbox",
        session_id=session_id,
        params={"url": "https://example.com"},
    )
    assert goto.ok is True
    assert goto.page_url == "https://example.com"
    assert goto.artifact_refs

    extract = await manager.execute_action(
        action="page.extract",
        effective_user_id="u1",
        provider_name="sandbox",
        session_id=session_id,
        params={"mode": "title"},
    )
    assert extract.ok is True
    assert extract.data["title"] == "Example"

    shot = await manager.execute_action(
        action="page.screenshot",
        effective_user_id="u1",
        provider_name="sandbox",
        session_id=session_id,
    )
    assert shot.ok is True
    assert shot.artifact_refs


@pytest.mark.asyncio
async def test_browser_tool_missing_action_returns_error(tmp_path) -> None:
    store = BrowserStore(tmp_path / "browser")
    manager = BrowserManager(
        config=_config(), store=store, providers={"sandbox": _FakeProvider()}
    )
    tool = BrowserTool(manager)

    result = await tool.execute({}, ToolContext(user_id="u1"))
    assert result.is_error is True
    assert "missing required field: action" in result.content


@pytest.mark.asyncio
async def test_browser_manager_rejects_cross_provider_session_id(tmp_path) -> None:
    store = BrowserStore(tmp_path / "browser")
    manager = BrowserManager(
        config=_config(),
        store=store,
        providers={"sandbox": _FakeProvider(), "kernel": _FakeKernelProvider()},
    )

    started = await manager.execute_action(
        action="session.start",
        effective_user_id="u1",
        provider_name="sandbox",
    )
    assert started.ok is True
    assert started.session_id is not None

    mismatch = await manager.execute_action(
        action="page.goto",
        effective_user_id="u1",
        provider_name="kernel",
        session_id=started.session_id,
        params={"url": "https://example.com"},
    )
    assert mismatch.ok is False
    assert mismatch.error_code == "session_not_found"
