"""Tests for sandboxed CLI memory commands."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ash_sandbox_cli.commands.memory import app
from typer.testing import CliRunner

from ash.config.models import AshConfig, ModelConfig
from ash.context_token import get_default_context_token_service
from ash.integrations import IntegrationContext, IntegrationRuntime, MemoryIntegration
from ash.integrations.rpc import active_rpc_server
from ash.memory.extractor import MemoryExtractor
from ash.sessions.types import session_key


def _runner(env: dict[str, str]) -> CliRunner:
    return CliRunner(env=env)


def _context_token(
    *,
    effective_user_id: str = "user-1",
    chat_id: str | None = "chat-1",
    chat_type: str | None = "group",
    provider: str | None = "telegram",
    thread_id: str | None = None,
    source_username: str | None = None,
    source_display_name: str | None = None,
    message_id: str | None = None,
    session_key: str | None = None,
    current_user_message: str | None = None,
) -> str:
    return get_default_context_token_service().issue(
        effective_user_id=effective_user_id,
        chat_id=chat_id,
        chat_type=chat_type,
        provider=provider,
        thread_id=thread_id,
        source_username=source_username,
        source_display_name=source_display_name,
        message_id=message_id,
        session_key=session_key,
        current_user_message=current_user_message,
    )


class TestMemoryExtract:
    """Tests for 'ash memory extract' command behavior."""

    def test_extract_uses_session_backed_rpc_with_explicit_session_coords(self):
        """When message_id is present, command should call memory.extract."""
        runner = _runner(
            {
                "ASH_CONTEXT_TOKEN": _context_token(
                    thread_id="thread-1",
                    source_username="alice",
                    source_display_name="Alice",
                    message_id="12345",
                    session_key="telegram_chat_1_thread_1",
                )
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
                "ASH_CONTEXT_TOKEN": _context_token(
                    source_username="alice",
                    source_display_name="Alice",
                    current_user_message="remember my todo list",
                )
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

    def test_extract_retries_with_explicit_message_when_session_lookup_misses(self):
        """When session lookup misses, command should retry using explicit message."""
        runner = _runner(
            {
                "ASH_CONTEXT_TOKEN": _context_token(
                    source_username="alice",
                    source_display_name="Alice",
                    message_id="missing-123",
                    current_user_message="remember i need to buy coffee",
                )
            }
        )

        with patch("ash_sandbox_cli.commands.memory.rpc_call") as mock_rpc:
            mock_rpc.side_effect = [
                {"stored": 0, "error": "Message not found in session"},
                {"stored": 1},
            ]
            result = runner.invoke(app, ["extract"])

        assert result.exit_code == 0
        assert "Extracted 1 memory(ies)" in result.stdout
        assert mock_rpc.call_count == 2
        first_call_args = mock_rpc.call_args_list[0][0]
        second_call_args = mock_rpc.call_args_list[1][0]
        assert first_call_args[0] == "memory.extract"
        assert second_call_args[0] == "memory.extract_from_messages"
        assert second_call_args[1]["messages"] == [
            {
                "role": "user",
                "content": "remember i need to buy coffee",
                "user_id": "user-1",
                "username": "alice",
                "display_name": "Alice",
            }
        ]

    @pytest.mark.asyncio
    async def test_extract_end_to_end_via_real_rpc_server(
        self, graph_store, tmp_path: Path
    ):
        """memory extract should work through real socket RPC integration wiring."""
        sessions_path = tmp_path / "sessions"
        sessions_path.mkdir(parents=True, exist_ok=True)

        key = session_key("telegram", "chat-1", "user-1", "thread-1")
        session_dir = sessions_path / key
        session_dir.mkdir(parents=True, exist_ok=True)

        external_id = "ext-msg-77"
        context_file = session_dir / "context.jsonl"
        entries = [
            {
                "type": "session",
                "id": "session-1",
                "created_at": datetime.now(UTC).isoformat(),
                "provider": "telegram",
                "user_id": "user-1",
                "chat_id": "chat-1",
                "version": "2",
            },
            {
                "type": "message",
                "id": "internal-msg-1",
                "role": "user",
                "content": "remember this integration test",
                "created_at": datetime.now(UTC).isoformat(),
                "token_count": 11,
                "username": "alice",
                "display_name": "Alice",
                "user_id": "user-1",
                "metadata": {"external_id": external_id},
            },
        ]
        context_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        extractor = MagicMock(spec=MemoryExtractor)
        extractor.extract_from_conversation = AsyncMock(return_value=[])
        extractor.classify_fact = AsyncMock(return_value=None)

        config = AshConfig(
            models={"default": ModelConfig(provider="openai", model="gpt-5-mini")}
        )
        components = SimpleNamespace(
            memory_manager=graph_store,
            memory_extractor=extractor,
        )
        context = IntegrationContext(
            config=config,
            components=components,  # type: ignore[arg-type]
            mode="chat",
            sessions_path=sessions_path,
        )
        runtime = IntegrationRuntime([MemoryIntegration()])

        socket_path = tmp_path / "rpc.sock"
        context_token = get_default_context_token_service().issue(
            effective_user_id="user-1",
            chat_id="chat-1",
            chat_type="group",
            provider="telegram",
            session_key=key,
            thread_id="thread-1",
            source_username="alice",
            source_display_name="Alice",
            message_id=external_id,
        )
        env = {
            "ASH_RPC_SOCKET": str(socket_path),
            "ASH_CONTEXT_TOKEN": context_token,
        }

        async with active_rpc_server(
            runtime=runtime,
            context=context,
            socket_path=socket_path,
        ):
            runner = _runner(env)
            result = await asyncio.to_thread(runner.invoke, app, ["extract"])

        assert result.exit_code == 0
        assert "No extractable facts found in this message." in result.stdout
