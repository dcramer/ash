from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import SecretStr

from ash.cli.runtime import bootstrap_runtime
from ash.config import AshConfig
from ash.config.models import ModelConfig, SentryConfig


@pytest.mark.asyncio
async def test_bootstrap_runtime_builds_workspace_and_components(
    monkeypatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    class _FakeLoader:
        def __init__(self, workspace_path: Path) -> None:
            calls["workspace_path"] = workspace_path

        def ensure_workspace(self) -> None:
            calls["ensured"] = True

        def load(self) -> Any:
            return SimpleNamespace(path=tmp_path / "workspace")

    async def _fake_create_agent(*, config, workspace, graph_dir, model_alias):
        calls["create_agent"] = {
            "config": config,
            "workspace": workspace,
            "graph_dir": graph_dir,
            "model_alias": model_alias,
        }
        return cast(Any, SimpleNamespace(agent=object()))

    monkeypatch.setattr("ash.cli.runtime.WorkspaceLoader", _FakeLoader)
    monkeypatch.setattr("ash.cli.runtime.create_agent", _fake_create_agent)
    monkeypatch.setattr("ash.cli.runtime.get_graph_dir", lambda: tmp_path / "graph")

    config = AshConfig(
        workspace=tmp_path / "workspace",
        models={"default": ModelConfig(provider="openai", model="gpt-5-mini")},
    )

    bootstrap = await bootstrap_runtime(
        config=config,
        model_alias="default",
        initialize_sentry=False,
    )

    assert calls["workspace_path"] == tmp_path / "workspace"
    assert calls["ensured"] is True
    assert calls["create_agent"]["graph_dir"] == tmp_path / "graph"
    assert calls["create_agent"]["model_alias"] == "default"
    assert bootstrap.sentry_initialized is False


@pytest.mark.asyncio
async def test_bootstrap_runtime_tracks_sentry_init(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("ash.cli.runtime.get_graph_dir", lambda: tmp_path / "graph")

    class _FakeLoader:
        def __init__(self, _workspace_path: Path) -> None:
            return None

        def ensure_workspace(self) -> None:
            return None

        def load(self) -> Any:
            return SimpleNamespace(path=tmp_path / "workspace")

    async def _fake_create_agent(*, config, workspace, graph_dir, model_alias):
        _ = (config, workspace, graph_dir, model_alias)
        return cast(Any, SimpleNamespace(agent=object()))

    monkeypatch.setattr("ash.cli.runtime.WorkspaceLoader", _FakeLoader)
    monkeypatch.setattr("ash.cli.runtime.create_agent", _fake_create_agent)
    monkeypatch.setattr("ash.observability.init_sentry", lambda *_args, **_kwargs: True)

    config = AshConfig(
        workspace=tmp_path / "workspace",
        models={"default": ModelConfig(provider="openai", model="gpt-5-mini")},
        sentry=SentryConfig(dsn=SecretStr("https://example.com/1")),
    )

    bootstrap = await bootstrap_runtime(
        config=config,
        initialize_sentry=True,
        sentry_server_mode=True,
    )

    assert bootstrap.sentry_initialized is True
