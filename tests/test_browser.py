from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ash.browser.manager import BrowserManager, create_browser_manager
from ash.browser.providers.base import (
    ProviderExtractResult,
    ProviderGotoResult,
    ProviderScreenshotResult,
    ProviderStartResult,
)
from ash.browser.store import BrowserStore
from ash.browser.types import BrowserSession
from ash.config.models import (
    AshConfig,
    BrowserConfig,
    BrowserSandboxConfig,
    ModelConfig,
)
from ash.tools.base import ToolContext
from ash.tools.builtin.browser import BrowserTool


class _FakeProvider:
    name = "sandbox"
    runs_in_sandbox_executor = True

    def __init__(self) -> None:
        self.closed_session_ids: list[str | None] = []
        self.last_goto_timeout_seconds: float | None = None
        self.last_wait_timeout_seconds: float | None = None

    async def start_session(self, *, session_id: str, profile_name: str | None):
        _ = profile_name
        return ProviderStartResult(provider_session_id=f"p-{session_id[:8]}")

    async def close_session(self, *, provider_session_id: str | None) -> None:
        self.closed_session_ids.append(provider_session_id)

    async def goto(
        self, *, provider_session_id: str | None, url: str, timeout_seconds: float
    ):
        _ = (provider_session_id, timeout_seconds)
        self.last_goto_timeout_seconds = timeout_seconds
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
        self.last_wait_timeout_seconds = timeout_seconds

    async def screenshot(self, *, provider_session_id: str | None):
        _ = provider_session_id
        return ProviderScreenshotResult(image_bytes=b"abc")


class _FakeKernelProvider(_FakeProvider):
    name = "kernel"


class _FakeHostRuntimeProvider(_FakeProvider):
    runs_in_sandbox_executor = False


class _FakeWarmupProvider(_FakeProvider):
    name = "sandbox"

    def __init__(self) -> None:
        self.warmup_calls = 0

    async def warmup(self) -> None:
        self.warmup_calls += 1


class _FailingRuntimeProvider(_FakeProvider):
    async def start_session(self, *, session_id: str, profile_name: str | None):
        _ = (session_id, profile_name)
        raise ValueError("sandbox_browser_launch_failed: cdp_not_ready")


def _config() -> AshConfig:
    return AshConfig(
        workspace=Path("tmp-workspace"),
        models={"default": ModelConfig(provider="openai", model="gpt-5.2")},
        browser=BrowserConfig(
            sandbox=BrowserSandboxConfig(
                runtime_required=False,
                runtime_mode="legacy",
            ),
        ),
    )


def _strict_config() -> AshConfig:
    return AshConfig(
        workspace=Path("tmp-workspace"),
        models={"default": ModelConfig(provider="openai", model="gpt-5.2")},
        browser=BrowserConfig(
            sandbox=BrowserSandboxConfig(
                runtime_required=True,
                runtime_mode="legacy",
            ),
        ),
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
async def test_browser_manager_clamps_action_timeouts(tmp_path) -> None:
    store = BrowserStore(tmp_path / "browser")
    provider = _FakeProvider()
    manager = BrowserManager(
        config=_config(), store=store, providers={"sandbox": provider}
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
        params={"url": "https://example.com", "timeout_seconds": 9999},
    )
    assert goto.ok is True
    assert provider.last_goto_timeout_seconds == 120.0

    waited = await manager.execute_action(
        action="page.wait_for",
        effective_user_id="u1",
        provider_name="sandbox",
        session_id=session_id,
        params={"selector": "#ready", "timeout_seconds": 0},
    )
    assert waited.ok is True
    assert provider.last_wait_timeout_seconds == 1.0


@pytest.mark.asyncio
async def test_browser_manager_runtime_failures_include_non_retry_guidance(
    tmp_path,
) -> None:
    store = BrowserStore(tmp_path / "browser")
    manager = BrowserManager(
        config=_config(),
        store=store,
        providers={"sandbox": _FailingRuntimeProvider()},
    )
    started = await manager.execute_action(
        action="session.start",
        effective_user_id="u1",
        provider_name="sandbox",
    )
    assert started.ok is False
    assert started.error_code == "action_failed"
    assert started.error_message is not None
    assert "Do NOT retry browser tool" in started.error_message


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


def test_browser_tool_provider_enum_reflects_manager_providers(tmp_path) -> None:
    store = BrowserStore(tmp_path / "browser")
    manager = BrowserManager(
        config=_config(), store=store, providers={"sandbox": _FakeProvider()}
    )
    tool = BrowserTool(manager)
    provider_schema = tool.input_schema["properties"]["provider"]
    assert provider_schema["enum"] == ["sandbox"]


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
    assert mismatch.error_code == "provider_not_supported"


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
    assert mismatch.error_code == "provider_not_supported"


@pytest.mark.asyncio
async def test_browser_manager_expires_stale_active_sessions(tmp_path) -> None:
    store = BrowserStore(tmp_path / "browser")
    cfg = _config()
    cfg.browser.max_session_minutes = 10
    provider = _FakeProvider()
    manager = BrowserManager(
        config=cfg,
        store=store,
        providers={"sandbox": provider},
    )

    stale_time = datetime.now(UTC) - timedelta(minutes=30)
    stale = BrowserSession(
        id="session-stale",
        name="stale",
        effective_user_id="u1",
        provider="sandbox",
        status="active",
        provider_session_id="p-stale",
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
    assert provider.closed_session_ids == ["p-stale"]


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


@pytest.mark.asyncio
async def test_browser_manager_requires_sandbox_runtime_when_enabled(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BrowserStore(tmp_path / "browser")
    manager = BrowserManager(
        config=_strict_config(),
        store=store,
        providers={"sandbox": _FakeHostRuntimeProvider()},
    )
    monkeypatch.setattr(manager, "_is_sandbox_runtime", lambda: False)

    started = await manager.execute_action(
        action="session.start",
        effective_user_id="u1",
        provider_name="sandbox",
    )
    assert started.ok is False
    assert started.error_code == "sandbox_runtime_required"


@pytest.mark.asyncio
async def test_browser_tool_respects_runtime_gate_for_agent_calls(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BrowserStore(tmp_path / "browser")
    manager = BrowserManager(
        config=_strict_config(),
        store=store,
        providers={"sandbox": _FakeHostRuntimeProvider()},
    )
    monkeypatch.setattr(manager, "_is_sandbox_runtime", lambda: False)
    tool = BrowserTool(manager)

    result = await tool.execute(
        {"action": "session.start", "provider": "sandbox"},
        ToolContext(user_id="u1"),
    )
    assert result.is_error is True
    assert "sandbox_runtime_required" in result.content


@pytest.mark.asyncio
async def test_browser_manager_allows_executor_offload_when_host_runtime(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BrowserStore(tmp_path / "browser")
    manager = BrowserManager(
        config=_strict_config(),
        store=store,  # Provider marks sandbox-executor offload as available.
        providers={"sandbox": _FakeProvider()},
    )
    monkeypatch.setattr(manager, "_is_sandbox_runtime", lambda: False)

    started = await manager.execute_action(
        action="session.start",
        effective_user_id="u1",
        provider_name="sandbox",
    )
    assert started.ok is True


@pytest.mark.asyncio
async def test_browser_manager_archive_closes_active_provider_session(tmp_path) -> None:
    store = BrowserStore(tmp_path / "browser")
    provider = _FakeProvider()
    manager = BrowserManager(
        config=_config(), store=store, providers={"sandbox": provider}
    )

    started = await manager.execute_action(
        action="session.start",
        effective_user_id="u1",
        session_name="work",
        provider_name="sandbox",
    )
    assert started.ok is True
    assert started.session_id is not None

    archived = await manager.execute_action(
        action="session.archive",
        effective_user_id="u1",
        provider_name="sandbox",
        session_id=started.session_id,
    )
    assert archived.ok is True
    assert len(provider.closed_session_ids) == 1
    assert provider.closed_session_ids[0] is not None


@pytest.mark.asyncio
async def test_browser_manager_warmup_default_provider(tmp_path) -> None:
    store = BrowserStore(tmp_path / "browser")
    provider = _FakeWarmupProvider()
    manager = BrowserManager(
        config=_config(), store=store, providers={"sandbox": provider}
    )
    await manager.warmup_default_provider()
    assert provider.warmup_calls == 1


@pytest.mark.asyncio
async def test_create_browser_manager_omits_kernel_when_unconfigured(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KERNEL_API_KEY", raising=False)
    cfg = AshConfig(
        workspace=tmp_path / "workspace",
        models={"default": ModelConfig(provider="openai", model="gpt-5.2")},
        browser=BrowserConfig(
            provider="sandbox",
            sandbox=BrowserSandboxConfig(
                runtime_required=False,
                runtime_mode="legacy",
            ),
            state_dir=tmp_path / "browser-state",
        ),
    )
    manager = create_browser_manager(cfg)

    attempted = await manager.execute_action(
        action="session.start",
        effective_user_id="u1",
        provider_name="kernel",
    )
    assert attempted.ok is False
    assert attempted.error_code == "invalid_provider"


def test_create_browser_manager_uses_kernel_when_configured(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KERNEL_API_KEY", raising=False)
    cfg = AshConfig(
        workspace=tmp_path / "workspace",
        models={"default": ModelConfig(provider="openai", model="gpt-5.2")},
        browser=BrowserConfig(
            provider="kernel",
            sandbox=BrowserSandboxConfig(
                runtime_required=False,
                runtime_mode="legacy",
            ),
            state_dir=tmp_path / "browser-state",
        ),
    )
    manager = create_browser_manager(cfg)

    assert manager.provider_names == ("kernel",)


def test_create_browser_manager_uses_configured_provider_even_with_kernel_api_key(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KERNEL_API_KEY", "test-key")
    cfg = AshConfig(
        workspace=tmp_path / "workspace",
        models={"default": ModelConfig(provider="openai", model="gpt-5.2")},
        browser=BrowserConfig(
            provider="sandbox",
            sandbox=BrowserSandboxConfig(
                runtime_required=False,
                runtime_mode="legacy",
            ),
            state_dir=tmp_path / "browser-state",
        ),
    )
    manager = create_browser_manager(cfg)
    assert manager.provider_names == ("sandbox",)


@pytest.mark.asyncio
async def test_browser_manager_kernel_page_actions_fail_fast(tmp_path) -> None:
    store = BrowserStore(tmp_path / "browser")
    manager = BrowserManager(config=_config(), store=store, providers={})
    result = await manager.execute_action(
        action="page.goto",
        effective_user_id="u1",
        provider_name="kernel",
        params={"url": "https://example.com"},
    )
    assert result.ok is False
    assert result.error_code == "invalid_provider"
