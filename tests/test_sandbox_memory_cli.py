"""Tests for sandboxed CLI memory commands."""

from unittest.mock import patch

from ash_sandbox_cli.commands.memory import app
from typer.testing import CliRunner


def _runner(env: dict[str, str]) -> CliRunner:
    return CliRunner(env=env)


class TestMemoryExtract:
    """Tests for 'ash memory extract' command behavior."""

    def test_extract_uses_session_backed_rpc_with_explicit_session_coords(self):
        """When message_id is present, command should call memory.extract."""
        runner = _runner(
            {
                "ASH_USER_ID": "user-1",
                "ASH_CHAT_ID": "chat-1",
                "ASH_CHAT_TYPE": "group",
                "ASH_THREAD_ID": "thread-1",
                "ASH_PROVIDER": "telegram",
                "ASH_USERNAME": "alice",
                "ASH_DISPLAY_NAME": "Alice",
                "ASH_MESSAGE_ID": "12345",
                "ASH_SESSION_KEY": "telegram_chat_1_thread_1",
            }
        )

        with patch("ash_sandbox_cli.commands.memory.rpc_call") as mock_rpc:
            mock_rpc.return_value = {"stored": 1}
            result = runner.invoke(app, ["extract"])

        assert result.exit_code == 0
        assert "Extracted 1 memory(ies)" in result.stdout
        mock_rpc.assert_called_once()
        call_args = mock_rpc.call_args[0]
        assert call_args[0] == "memory.extract"
        params = call_args[1]
        assert params["provider"] == "telegram"
        assert params["user_id"] == "user-1"
        assert params["chat_id"] == "chat-1"
        assert params["chat_type"] == "group"
        assert params["thread_id"] == "thread-1"
        assert params["message_id"] == "12345"
        assert params["session_key"] == "telegram_chat_1_thread_1"

    def test_extract_falls_back_to_explicit_messages_when_no_message_id(self):
        """When message_id is absent, command should call memory.extract_from_messages."""
        runner = _runner(
            {
                "ASH_USER_ID": "user-1",
                "ASH_CHAT_ID": "chat-1",
                "ASH_CHAT_TYPE": "group",
                "ASH_PROVIDER": "telegram",
                "ASH_USERNAME": "alice",
                "ASH_DISPLAY_NAME": "Alice",
                "ASH_CURRENT_USER_MESSAGE": "remember my todo list",
            }
        )

        with patch("ash_sandbox_cli.commands.memory.rpc_call") as mock_rpc:
            mock_rpc.return_value = {"stored": 2}
            result = runner.invoke(app, ["extract"])

        assert result.exit_code == 0
        assert "Extracted 2 memory(ies)" in result.stdout
        mock_rpc.assert_called_once()
        call_args = mock_rpc.call_args[0]
        assert call_args[0] == "memory.extract_from_messages"
        params = call_args[1]
        assert params["provider"] == "telegram"
        assert params["user_id"] == "user-1"
        assert params["chat_id"] == "chat-1"
        assert params["chat_type"] == "group"
        assert params["source_username"] == "alice"
        assert params["source_display_name"] == "Alice"
        assert params["messages"] == [
            {
                "role": "user",
                "content": "remember my todo list",
                "user_id": "user-1",
                "username": "alice",
                "display_name": "Alice",
            }
        ]
