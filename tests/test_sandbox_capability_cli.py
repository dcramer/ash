"""Tests for sandboxed capability CLI commands."""

from unittest.mock import patch

import pytest
from ash_sandbox_cli.commands.capability import app
from typer.testing import CliRunner

from ash.context_token import get_default_context_token_service


def _context_token(
    *,
    effective_user_id: str = "user123",
    chat_id: str | None = "chat456",
    chat_type: str | None = "private",
    provider: str | None = "telegram",
) -> str:
    return get_default_context_token_service().issue(
        effective_user_id=effective_user_id,
        chat_id=chat_id,
        chat_type=chat_type,
        provider=provider,
    )


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner(env={"ASH_CONTEXT_TOKEN": _context_token()})


@pytest.fixture
def mock_rpc():
    with patch("ash_sandbox_cli.commands.capability.rpc_call") as mock:
        yield mock


def test_list_capabilities(cli_runner: CliRunner, mock_rpc) -> None:
    mock_rpc.return_value = {
        "capabilities": [
            {
                "id": "gog.email",
                "description": "Email operations",
                "available": True,
                "authenticated": False,
                "operations": ["list_messages"],
            }
        ]
    }

    result = cli_runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "gog.email" in result.stdout
    assert "Available: yes" in result.stdout
    assert "Authenticated: no" in result.stdout


def test_invoke_capability(cli_runner: CliRunner, mock_rpc) -> None:
    mock_rpc.return_value = {
        "ok": True,
        "request_id": "cap_123",
        "output": {"status": "ok"},
    }

    result = cli_runner.invoke(
        app,
        [
            "invoke",
            "--capability",
            "gog.email",
            "--operation",
            "list_messages",
            "--input-json",
            '{"folder":"inbox"}',
        ],
    )
    assert result.exit_code == 0
    assert "Capability invocation succeeded" in result.stdout
    mock_rpc.assert_called_once()
    assert mock_rpc.call_args[0][0] == "capability.invoke"
    params = mock_rpc.call_args[0][1]
    assert params["capability"] == "gog.email"
    assert params["operation"] == "list_messages"
    assert params["input"] == {"folder": "inbox"}


def test_invoke_rejects_non_object_json(cli_runner: CliRunner, mock_rpc) -> None:
    result = cli_runner.invoke(
        app,
        [
            "invoke",
            "--capability",
            "gog.email",
            "--operation",
            "list_messages",
            "--input-json",
            '["bad"]',
        ],
    )
    assert result.exit_code == 1
    assert "must decode to an object" in result.output
    mock_rpc.assert_not_called()


def test_auth_begin(cli_runner: CliRunner, mock_rpc) -> None:
    mock_rpc.return_value = {
        "flow_id": "caf_1",
        "auth_url": "https://auth.ash.invalid",
        "expires_at": "2026-02-24T20:10:00Z",
    }

    result = cli_runner.invoke(
        app,
        ["auth", "begin", "--capability", "gog.email", "--account", "work"],
    )
    assert result.exit_code == 0
    assert "Started capability auth flow" in result.stdout
    assert "caf_1" in result.stdout
    assert mock_rpc.call_args[0][0] == "capability.auth.begin"


def test_auth_complete_requires_code_or_callback(
    cli_runner: CliRunner, mock_rpc
) -> None:
    result = cli_runner.invoke(app, ["auth", "complete", "--flow-id", "caf_1"])
    assert result.exit_code == 1
    assert "Must specify either --callback-url or --code" in result.output
    mock_rpc.assert_not_called()


def test_auth_complete(cli_runner: CliRunner, mock_rpc) -> None:
    mock_rpc.return_value = {"ok": True, "account_ref": "work"}
    result = cli_runner.invoke(
        app,
        [
            "auth",
            "complete",
            "--flow-id",
            "caf_1",
            "--callback-url",
            "https://localhost/callback?code=abc",
        ],
    )
    assert result.exit_code == 0
    assert "Capability auth completed" in result.stdout
    assert "account_ref=work" in result.stdout
