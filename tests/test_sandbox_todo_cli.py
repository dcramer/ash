"""Tests for sandboxed CLI todo commands."""

from __future__ import annotations

from unittest.mock import patch

from ash_sandbox_cli.commands.todo import app
from typer.testing import CliRunner

from ash.context_token import get_default_context_token_service


def _context_token(
    *,
    effective_user_id: str = "user-1",
    chat_id: str | None = "chat-1",
    provider: str | None = "telegram",
    timezone: str | None = "UTC",
) -> str:
    return get_default_context_token_service().issue(
        effective_user_id=effective_user_id,
        chat_id=chat_id,
        provider=provider,
        timezone=timezone,
    )


def _runner() -> CliRunner:
    return CliRunner(
        env={
            "ASH_CONTEXT_TOKEN": _context_token(),
        }
    )


def test_todo_add_calls_rpc() -> None:
    runner = _runner()
    with patch("ash_sandbox_cli.commands.todo.rpc_call") as mock_rpc:
        mock_rpc.return_value = {
            "todo": {"id": "abc12345", "content": "buy milk", "status": "open"}
        }
        result = runner.invoke(app, ["add", "buy milk"])

    assert result.exit_code == 0
    assert "Created todo" in result.stdout
    mock_rpc.assert_called_once()
    assert mock_rpc.call_args[0][0] == "todo.create"


def test_todo_list_calls_rpc() -> None:
    runner = _runner()
    with patch("ash_sandbox_cli.commands.todo.rpc_call") as mock_rpc:
        mock_rpc.return_value = [
            {"id": "abc12345", "content": "buy milk", "status": "open"},
            {"id": "def67890", "content": "file taxes", "status": "done"},
        ]
        result = runner.invoke(app, ["list", "--include-done"])

    assert result.exit_code == 0
    assert "Total: 2 todo(s)" in result.stdout
    mock_rpc.assert_called_once()
    assert mock_rpc.call_args[0][0] == "todo.list"


def test_todo_done_calls_rpc() -> None:
    runner = _runner()
    with patch("ash_sandbox_cli.commands.todo.rpc_call") as mock_rpc:
        mock_rpc.return_value = {
            "todo": {"id": "abc12345", "content": "buy milk", "status": "done"}
        }
        result = runner.invoke(app, ["done", "--id", "abc12345"])

    assert result.exit_code == 0
    assert "Completed todo" in result.stdout
    assert mock_rpc.call_args[0][0] == "todo.complete"


def test_todo_list_all_sets_include_flags() -> None:
    runner = _runner()
    with patch("ash_sandbox_cli.commands.todo.rpc_call") as mock_rpc:
        mock_rpc.return_value = []
        result = runner.invoke(app, ["list", "--all"])

    assert result.exit_code == 0
    method, params = mock_rpc.call_args[0]
    assert method == "todo.list"
    assert params["include_done"] is True
    assert params["include_deleted"] is True


def test_todo_remind_calls_update_rpc() -> None:
    runner = _runner()
    with patch("ash_sandbox_cli.commands.todo.rpc_call") as mock_rpc:
        mock_rpc.return_value = {
            "todo": {
                "id": "abc12345",
                "content": "buy milk",
                "status": "open",
                "linked_schedule_entry_id": "sched1234",
            }
        }
        result = runner.invoke(
            app,
            ["remind", "--id", "abc12345", "--at", "2026-03-01T10:00:00+00:00"],
        )

    assert result.exit_code == 0
    method, params = mock_rpc.call_args[0]
    assert method == "todo.update"
    assert params["todo_id"] == "abc12345"
    assert params["reminder_at"] is not None


def test_todo_unremind_calls_update_rpc() -> None:
    runner = _runner()
    with patch("ash_sandbox_cli.commands.todo.rpc_call") as mock_rpc:
        mock_rpc.return_value = {"todo": {"id": "abc12345"}}
        result = runner.invoke(app, ["unremind", "--id", "abc12345"])

    assert result.exit_code == 0
    method, params = mock_rpc.call_args[0]
    assert method == "todo.update"
    assert params["todo_id"] == "abc12345"
    assert params["clear_reminder"] is True


def test_todo_edit_calls_update_rpc() -> None:
    runner = _runner()
    with patch("ash_sandbox_cli.commands.todo.rpc_call") as mock_rpc:
        mock_rpc.return_value = {
            "todo": {"id": "abc12345", "content": "buy oat milk", "status": "open"}
        }
        result = runner.invoke(
            app,
            ["edit", "--id", "abc12345", "--text", "buy oat milk"],
        )

    assert result.exit_code == 0
    method, params = mock_rpc.call_args[0]
    assert method == "todo.update"
    assert params["todo_id"] == "abc12345"
    assert params["content"] == "buy oat milk"


def test_todo_undone_calls_uncomplete_rpc() -> None:
    runner = _runner()
    with patch("ash_sandbox_cli.commands.todo.rpc_call") as mock_rpc:
        mock_rpc.return_value = {
            "todo": {"id": "abc12345", "content": "buy milk", "status": "open"}
        }
        result = runner.invoke(app, ["undone", "--id", "abc12345"])

    assert result.exit_code == 0
    assert "Reopened todo" in result.stdout
    assert mock_rpc.call_args[0][0] == "todo.uncomplete"


def test_todo_delete_calls_rpc() -> None:
    runner = _runner()
    with patch("ash_sandbox_cli.commands.todo.rpc_call") as mock_rpc:
        mock_rpc.return_value = {
            "todo": {"id": "abc12345", "content": "buy milk", "status": "open"}
        }
        result = runner.invoke(app, ["delete", "--id", "abc12345"])

    assert result.exit_code == 0
    assert "Deleted todo" in result.stdout
    assert mock_rpc.call_args[0][0] == "todo.delete"


def test_todo_edit_rejects_unparseable_due_time() -> None:
    runner = _runner()
    with patch("ash_sandbox_cli.commands.todo.rpc_call") as mock_rpc:
        result = runner.invoke(
            app,
            ["edit", "--id", "abc12345", "--due", "not-a-time"],
        )

    assert result.exit_code == 1
    assert "Could not parse due time" in result.stderr
    mock_rpc.assert_not_called()


def test_todo_remind_requires_exactly_one_trigger() -> None:
    runner = _runner()
    with patch("ash_sandbox_cli.commands.todo.rpc_call") as mock_rpc:
        result = runner.invoke(app, ["remind", "--id", "abc12345"])

    assert result.exit_code == 1
    assert "Must specify exactly one of --at or --cron" in result.stderr
    mock_rpc.assert_not_called()
