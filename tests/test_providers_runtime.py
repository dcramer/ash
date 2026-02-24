from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import SecretStr

from ash.config import AshConfig
from ash.config.models import ModelConfig, TelegramConfig
from ash.providers.runtime import build_provider_runtime
from ash.scheduling.types import ScheduleEntry


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
    assert runtime.persisters == {}


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
    assert "telegram" in runtime.persisters
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


@pytest.mark.asyncio
async def test_telegram_persister_writes_session_and_history(monkeypatch) -> None:
    session_calls: list[dict[str, Any]] = []
    history_calls: list[dict[str, Any]] = []

    class _FakeSessionManager:
        def __init__(self, *, provider, chat_id, user_id, thread_id) -> None:
            session_calls.append(
                {
                    "provider": provider,
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "thread_id": thread_id,
                }
            )

        async def add_assistant_message(
            self, *, content, token_count, metadata
        ) -> None:
            session_calls.append(
                {
                    "content": content,
                    "token_count": token_count,
                    "metadata": metadata,
                }
            )

    class _FakeChatHistoryWriter:
        def __init__(self, provider: str, chat_id: str) -> None:
            history_calls.append({"provider": provider, "chat_id": chat_id})

        def record_bot_message(self, *, content, metadata) -> None:
            history_calls.append({"content": content, "metadata": metadata})

    class _FakeTelegramProvider:
        def __init__(self, **_kwargs) -> None:
            return None

        async def send_message(
            self, _chat_id: str, _text: str, *, reply_to: str | None = None
        ) -> str:
            _ = reply_to
            return "sent-id"

    monkeypatch.setattr("ash.providers.runtime.TelegramProvider", _FakeTelegramProvider)
    monkeypatch.setattr("ash.sessions.SessionManager", _FakeSessionManager)
    monkeypatch.setattr("ash.chats.ChatHistoryWriter", _FakeChatHistoryWriter)
    monkeypatch.setattr("ash.core.tokens.estimate_tokens", lambda text: len(text))

    config = AshConfig(
        workspace=Path("workspace"),
        models={"default": ModelConfig(provider="openai", model="gpt-5-mini")},
        telegram=TelegramConfig(bot_token=SecretStr("token")),
    )

    runtime = build_provider_runtime(config)
    persister = runtime.persisters["telegram"]
    entry = ScheduleEntry(
        message="scheduled", provider="telegram", chat_id="123", user_id="456"
    )

    await persister(entry, "hello", "mid-1")

    assert session_calls[0] == {
        "provider": "telegram",
        "chat_id": "123",
        "user_id": "456",
        "thread_id": "mid-1",
    }
    assert session_calls[1]["content"] == "hello"
    assert session_calls[1]["metadata"] == {"external_id": "mid-1"}
    assert history_calls[0] == {"provider": "telegram", "chat_id": "123"}
    assert history_calls[1]["content"] == "hello"
    assert history_calls[1]["metadata"] == {
        "external_id": "mid-1",
        "thread_id": "mid-1",
    }
