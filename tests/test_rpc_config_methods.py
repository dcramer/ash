from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from ash.config.models import AshConfig, ModelConfig
from ash.rpc.methods.config import register_config_methods


class _FakeRPCServer:
    def __init__(self) -> None:
        self.methods: dict[
            str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
        ] = {}

    def register(self, name: str, handler) -> None:  # noqa: ANN001
        self.methods[name] = handler


@pytest.mark.asyncio
async def test_config_reload_reloads_all_skill_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
[models.default]
provider = "openai"
model = "gpt-5.2"

[skills.demo]
enabled = true
API_KEY = "x"
"""
    )
    monkeypatch.setattr("ash.rpc.methods.config.get_config_path", lambda: config_file)

    config = AshConfig(
        workspace=tmp_path,
        models={"default": ModelConfig(provider="openai", model="gpt-5.2")},
    )
    skill_registry = MagicMock()

    server = _FakeRPCServer()
    register_config_methods(cast(Any, server), config, skill_registry)

    handler = server.methods["config.reload"]
    result = await handler({})

    assert result["success"] is True
    skill_registry.reload_all.assert_called_once_with(tmp_path)
