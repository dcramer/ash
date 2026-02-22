from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
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
from ash.browser.types import BrowserSession
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


@pytest.mark.asyncio
async def test_browser_manager_no_cross_provider_fallback_without_session_ref(
    tmp_path,
) -> None:
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

    mismatch = await manager.execute_action(
        action="page.goto",
        effective_user_id="u1",
        provider_name="kernel",
        params={"url": "https://example.com"},
    )
    assert mismatch.ok is False
    assert mismatch.error_code == "session_not_found"


@pytest.mark.asyncio
async def test_browser_manager_expires_stale_active_sessions(tmp_path) -> None:
    store = BrowserStore(tmp_path / "browser")
    cfg = _config()
    cfg.browser.max_session_minutes = 10
    manager = BrowserManager(
        config=cfg,
        store=store,
        providers={"sandbox": _FakeProvider()},
    )

    stale_time = datetime.now(UTC) - timedelta(minutes=30)
    stale = BrowserSession(
        id="session-stale",
        name="stale",
        effective_user_id="u1",
        provider="sandbox",
        status="active",
        updated_at=stale_time,
        created_at=stale_time,
    )
    store.append_session(stale)

    listed = await manager.execute_action(
        action="session.list",
        effective_user_id="u1",
        provider_name="sandbox",
    )
    assert listed.ok is True

    refreshed = store.get_session("session-stale")
    assert refreshed is not None
    assert refreshed.status == "closed"
    assert refreshed.last_error == "session_expired"


@pytest.mark.asyncio
async def test_browser_manager_prunes_expired_artifacts(tmp_path) -> None:
    store = BrowserStore(tmp_path / "browser")
    cfg = _config()
    cfg.browser.artifacts_retention_days = 1
    manager = BrowserManager(
        config=cfg,
        store=store,
        providers={"sandbox": _FakeProvider()},
    )

    old_dir = store.artifacts_dir / "old-session"
    old_dir.mkdir(parents=True, exist_ok=True)
    old_file = old_dir / "old.html"
    old_file.write_text("old")
    old_ts = (datetime.now(UTC) - timedelta(days=3)).timestamp()
    os.utime(old_file, (old_ts, old_ts))

    new_dir = store.artifacts_dir / "new-session"
    new_dir.mkdir(parents=True, exist_ok=True)
    new_file = new_dir / "new.html"
    new_file.write_text("new")

    listed = await manager.execute_action(
        action="session.list",
        effective_user_id="u1",
        provider_name="sandbox",
    )
    assert listed.ok is True

    assert not old_file.exists()
    assert new_file.exists()
