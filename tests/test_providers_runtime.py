from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import SecretStr

from ash.config import AshConfig
from ash.config.models import ModelConfig, TelegramConfig
from ash.providers.runtime import build_provider_runtime


def _config(tmp_path: Path, telegram: TelegramConfig | None = None) -> AshConfig:
    return AshConfig(
        workspace=tmp_path / "workspace",
        models={"default": ModelConfig(provider="openai", model="gpt-5-mini")},
        telegram=telegram,
    )


def test_build_provider_runtime_without_telegram(tmp_path: Path) -> None:
    runtime = build_provider_runtime(_config(tmp_path))

    assert runtime.telegram_provider is None
    assert runtime.senders == {}
    assert runtime.registrars == {}


def test_build_provider_runtime_with_telegram(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, Any] = {}

    class _FakeTelegramProvider:
        def __init__(
            self,
            *,
            bot_token: str,
            allowed_users,
            allowed_groups,
            group_mode,
            passive_config,
        ) -> None:
            calls["init"] = {
                "bot_token": bot_token,
                "allowed_users": allowed_users,
                "allowed_groups": allowed_groups,
                "group_mode": group_mode,
                "passive_config": passive_config,
            }

        async def send_message(
            self, _chat_id: str, _text: str, *, reply_to: str | None = None
        ) -> str:
            _ = reply_to
            return "sent-id"

    monkeypatch.setattr("ash.providers.runtime.TelegramProvider", _FakeTelegramProvider)

    config = _config(
        tmp_path,
        telegram=TelegramConfig(
            bot_token=SecretStr("token"),
            allowed_users=["foo"],
            allowed_groups=["bar"],
            group_mode="always",
        ),
    )
    runtime = build_provider_runtime(config)

    assert calls["init"]["bot_token"] == "token"
    assert "telegram" in runtime.senders
    assert "telegram" in runtime.registrars
    assert runtime.telegram_provider is not None


@pytest.mark.asyncio
async def test_telegram_registrar_wires_thread_index(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class _FakeTelegramProvider:
        def __init__(self, **_kwargs) -> None:
            return None

        async def send_message(
            self, _chat_id: str, _text: str, *, reply_to: str | None = None
        ) -> str:
            _ = reply_to
            return "sent-id"

    class _FakeThreadIndex:
        def __init__(self, _manager) -> None:
            return None

        def register_message(self, external_id: str, thread_id: str) -> None:
            calls.append((external_id, thread_id))

    monkeypatch.setattr(
        "ash.chats.ChatStateManager",
        lambda provider, chat_id: cast(
            Any, SimpleNamespace(provider=provider, chat_id=chat_id)
        ),
    )
    monkeypatch.setattr("ash.providers.runtime.TelegramProvider", _FakeTelegramProvider)
    monkeypatch.setattr("ash.chats.ThreadIndex", _FakeThreadIndex)

    config = AshConfig(
        workspace=Path("workspace"),
        models={"default": ModelConfig(provider="openai", model="gpt-5-mini")},
        telegram=TelegramConfig(bot_token=SecretStr("token")),
    )

    runtime = build_provider_runtime(config)
    registrar = runtime.registrars["telegram"]
    await registrar("123", "456")

    assert calls == [("456", "456")]
