from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
import typer

from ash.cli.commands.chat import (
    _new_cli_session_state,
    _resolve_model_alias,
    _validate_model_alias,
    _validate_model_credentials,
)
from ash.config import AshConfig
from ash.config.models import ModelConfig


def _config(provider: Literal["anthropic", "openai", "openai-oauth"]) -> AshConfig:
    return AshConfig(
        workspace=Path("tmp-workspace"),
        models={"default": ModelConfig(provider=provider, model="test-model")},
    )


def test_resolve_model_alias_prefers_cli_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASH_MODEL", "from-env")
    assert _resolve_model_alias("from-cli") == "from-cli"


def test_resolve_model_alias_uses_env_when_cli_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASH_MODEL", "from-env")
    assert _resolve_model_alias(None) == "from-env"


def test_resolve_model_alias_defaults_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ASH_MODEL", raising=False)
    assert _resolve_model_alias(None) == "default"


def test_validate_model_alias_raises_on_unknown_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    monkeypatch.setattr("ash.cli.commands.chat.error", lambda msg: messages.append(msg))

    with pytest.raises(typer.Exit):
        _validate_model_alias(_config("openai"), "missing")

    assert len(messages) == 1
    assert "Unknown model alias 'missing'" in messages[0]


def test_validate_model_credentials_raises_when_oauth_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    monkeypatch.setattr("ash.cli.commands.chat.error", lambda msg: messages.append(msg))
    cfg = _config("openai-oauth")

    with pytest.raises(typer.Exit):
        _validate_model_credentials(cfg, "default")

    assert messages == [
        "No OAuth credentials for openai-oauth. Run 'ash auth login' first."
    ]


def test_validate_model_credentials_raises_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    monkeypatch.setattr("ash.cli.commands.chat.error", lambda msg: messages.append(msg))
    cfg = _config("anthropic")

    with pytest.raises(typer.Exit):
        _validate_model_credentials(cfg, "default")

    assert len(messages) == 1
    assert "No API key for provider 'anthropic'" in messages[0]
    assert "ANTHROPIC_API_KEY" in messages[0]


def test_validate_model_credentials_passes_when_api_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = _config("openai")

    _validate_model_credentials(cfg, "default")


def test_new_cli_session_state_sets_private_chat_context() -> None:
    session = _new_cli_session_state("sess-test")

    assert session.provider == "cli"
    assert session.chat_id == "local"
    assert session.user_id == "local-user"
    assert session.context.chat_type == "private"
    assert session.context.chat_title == "Ash CLI"
