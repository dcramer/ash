"""Tests for provider implementations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ash.providers.base import IncomingMessage, OutgoingMessage
from ash.providers.telegram.handlers import TelegramMessageHandler
from ash.providers.telegram.provider import TelegramProvider


class TestTelegramProvider:
    """Tests for TelegramProvider."""

    @pytest.fixture
    def provider(self):
        """Create a Telegram provider with mock bot."""
        with patch("ash.providers.telegram.provider.Bot") as mock_bot_class:
            mock_bot = MagicMock()
            mock_bot.send_message = AsyncMock()
            mock_bot.send_chat_action = AsyncMock()
            mock_bot.edit_message_text = AsyncMock()
            mock_bot.delete_message = AsyncMock()
            mock_bot.delete_webhook = AsyncMock()
            mock_bot.session = MagicMock()
            mock_bot.session.close = AsyncMock()
            mock_bot_class.return_value = mock_bot

            provider = TelegramProvider(
                bot_token="test_token",
                allowed_users=["@testuser", "12345"],
            )
            provider._bot = mock_bot
            yield provider

    def test_name(self, provider):
        """Test provider name."""
        assert provider.name == "telegram"

    def test_is_user_allowed_by_id(self, provider):
        """Test user allowed by ID."""
        assert provider._is_user_allowed(12345, None) is True
        assert provider._is_user_allowed(99999, None) is False

    def test_is_user_allowed_by_username(self, provider):
        """Test user allowed by username."""
        assert provider._is_user_allowed(0, "testuser") is True
        assert provider._is_user_allowed(0, "otheruser") is False

    def test_is_user_allowed_empty_list(self):
        """Test all users allowed when list is empty."""
        with patch("ash.providers.telegram.provider.Bot"):
            provider = TelegramProvider(bot_token="test", allowed_users=[])
            assert provider._is_user_allowed(12345, "anyone") is True

    async def test_send_message(self, provider):
        """Test sending a message."""
        provider._bot.send_message.return_value = MagicMock(message_id=123)

        message = OutgoingMessage(
            chat_id="456",
            text="Hello, world!",
        )
        msg_id = await provider.send(message)

        assert msg_id == "123"
        provider._bot.send_message.assert_called_once()
        call_kwargs = provider._bot.send_message.call_args.kwargs
        assert call_kwargs["chat_id"] == 456
        assert call_kwargs["text"] == "Hello, world!"

    async def test_send_typing(self, provider):
        """Test sending typing indicator."""
        await provider.send_typing("456")

        provider._bot.send_chat_action.assert_called_once_with(
            chat_id=456,
            action="typing",
        )

    async def test_delete_message(self, provider):
        """Test deleting a message."""
        await provider.delete("456", "123")

        provider._bot.delete_message.assert_called_once_with(
            chat_id=456,
            message_id=123,
        )


class TestTelegramMessageHandler:
    """Tests for TelegramMessageHandler."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = MagicMock()
        provider.name = "telegram"
        provider.send = AsyncMock(return_value="123")  # Returns message ID
        provider.send_streaming = AsyncMock(return_value="123")
        provider.send_typing = AsyncMock()
        provider.set_reaction = AsyncMock()
        provider.clear_reaction = AsyncMock()
        return provider

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = MagicMock()
        agent.process_message = AsyncMock(
            return_value=MagicMock(
                text="Response from agent", compaction=None, tool_calls=[]
            )
        )

        async def mock_stream():
            yield "Response "
            yield "from "
            yield "agent"

        agent.process_message_streaming = MagicMock(return_value=mock_stream())
        return agent

    @pytest.fixture
    async def handler(self, mock_provider, mock_agent, database, tmp_path):
        """Create a message handler with temp sessions path."""
        handler = TelegramMessageHandler(
            provider=mock_provider,
            agent=mock_agent,
            database=database,
            streaming=True,
        )
        # Store tmp_path for tests to use
        handler._test_sessions_path = tmp_path  # type: ignore[attr-defined]
        return handler

    @pytest.fixture
    def incoming_message(self):
        """Create an incoming message."""
        return IncomingMessage(
            id="1",
            chat_id="456",
            user_id="789",
            text="Hello!",
            username="testuser",
            display_name="Test User",
        )

    async def test_handle_message_sends_typing(
        self, handler, mock_provider, mock_agent, incoming_message
    ):
        """Test that handling a message sends typing indicator."""
        from ash.sessions import SessionManager

        # Set up session manager to use temp path
        session_manager = SessionManager(
            provider="telegram",
            chat_id="456",
            user_id="789",
            sessions_path=handler._test_sessions_path,
        )
        handler._session_managers[session_manager.session_key] = session_manager

        # Create fresh async generator for this test
        async def mock_stream():
            yield "Response"

        mock_agent.process_message_streaming = MagicMock(return_value=mock_stream())

        await handler.handle_message(incoming_message)

        mock_provider.send_typing.assert_called_once_with("456")

    async def test_handle_message_streaming(
        self, handler, mock_provider, mock_agent, incoming_message
    ):
        """Test handling message with streaming response.

        New behavior: fast responses (<5s) are accumulated and sent as single
        message, not streamed via send_streaming().
        """
        from ash.sessions import SessionManager

        # Set up session manager to use temp path
        session_manager = SessionManager(
            provider="telegram",
            chat_id="456",
            user_id="789",
            sessions_path=handler._test_sessions_path,
        )
        handler._session_managers[session_manager.session_key] = session_manager

        # Create fresh async generator for this test
        async def mock_stream():
            yield "Response "
            yield "from "
            yield "agent"

        mock_agent.process_message_streaming = MagicMock(return_value=mock_stream())

        await handler.handle_message(incoming_message)

        # Fast responses are accumulated and sent as single message
        mock_provider.send.assert_called()
        # Get the last call (final response)
        call_args = mock_provider.send.call_args
        assert call_args[0][0].chat_id == "456"
        assert call_args[0][0].text == "Response from agent"
        assert call_args[0][0].reply_to_message_id == "1"

    async def test_handle_message_non_streaming(
        self, mock_provider, mock_agent, database, incoming_message, tmp_path
    ):
        """Test handling message with non-streaming response."""
        from ash.sessions import SessionManager

        handler = TelegramMessageHandler(
            provider=mock_provider,
            agent=mock_agent,
            database=database,
            streaming=False,
        )

        # Set up session manager to use temp path
        session_manager = SessionManager(
            provider="telegram",
            chat_id="456",
            user_id="789",
            sessions_path=tmp_path,
        )
        handler._session_managers[session_manager.session_key] = session_manager

        await handler.handle_message(incoming_message)

        mock_agent.process_message.assert_called_once()
        mock_provider.send.assert_called_once()

    async def test_session_creation(self, handler, incoming_message):
        """Test session is created for new chat."""
        from ash.sessions import SessionManager

        # Set up session manager to use temp path
        session_manager = SessionManager(
            provider="telegram",
            chat_id="456",
            user_id="789",
            sessions_path=handler._test_sessions_path,
        )
        handler._session_managers[session_manager.session_key] = session_manager

        session = await handler._get_or_create_session(incoming_message)

        assert session.chat_id == "456"
        assert session.user_id == "789"
        assert session.provider == "telegram"

    async def test_session_reuse(self, handler, incoming_message):
        """Test session is reused for same chat."""
        from ash.sessions import SessionManager

        # Set up session manager to use temp path
        session_manager = SessionManager(
            provider="telegram",
            chat_id="456",
            user_id="789",
            sessions_path=handler._test_sessions_path,
        )
        handler._session_managers[session_manager.session_key] = session_manager

        session1 = await handler._get_or_create_session(incoming_message)
        session2 = await handler._get_or_create_session(incoming_message)

        assert session1 is session2

    async def test_session_restoration(self, handler, incoming_message, tmp_path):
        """Test messages are restored from JSONL files."""
        from ash.sessions import SessionManager

        # Pre-populate JSONL session files
        session_manager = SessionManager(
            provider="telegram",
            chat_id="456",
            user_id="789",
            sessions_path=tmp_path,
        )
        await session_manager.ensure_session()
        await session_manager.add_user_message(
            content="Previous message",
            token_count=10,
        )
        await session_manager.add_assistant_message(
            content="Previous response",
            token_count=10,
        )

        # Override the handler's session manager cache to use our temp path
        handler._session_managers[session_manager.session_key] = session_manager

        # Get session - should restore messages from JSONL
        session = await handler._get_or_create_session(incoming_message)

        assert len(session.messages) == 2
        # Messages are in LLM format (Message objects)
        assert session.messages[0].content == "Previous message"
        assert session.messages[1].content == "Previous response"

    async def test_clear_session(self, handler, incoming_message):
        """Test clearing a session."""
        from ash.sessions import SessionManager

        # Set up session manager to use temp path
        session_manager = SessionManager(
            provider="telegram",
            chat_id="456",
            user_id="789",
            sessions_path=handler._test_sessions_path,
        )
        handler._session_managers[session_manager.session_key] = session_manager

        await handler._get_or_create_session(incoming_message)
        assert len(handler._sessions) == 1

        handler.clear_session("456")
        assert len(handler._sessions) == 0

    async def test_clear_all_sessions(self, handler, incoming_message):
        """Test clearing all sessions."""
        from ash.sessions import SessionManager

        # Set up session manager for first message
        session_manager1 = SessionManager(
            provider="telegram",
            chat_id="456",
            user_id="789",
            sessions_path=handler._test_sessions_path,
        )
        handler._session_managers[session_manager1.session_key] = session_manager1

        await handler._get_or_create_session(incoming_message)

        # Create another session
        msg2 = IncomingMessage(
            id="2",
            chat_id="999",
            user_id="888",
            text="Hi",
        )
        session_manager2 = SessionManager(
            provider="telegram",
            chat_id="999",
            user_id="888",
            sessions_path=handler._test_sessions_path,
        )
        handler._session_managers[session_manager2.session_key] = session_manager2

        await handler._get_or_create_session(msg2)
        assert len(handler._sessions) == 2

        handler.clear_all_sessions()
        assert len(handler._sessions) == 0

    async def test_message_persistence(self, handler, incoming_message, tmp_path):
        """Test messages are persisted to JSONL files."""
        from ash.sessions import SessionManager, SessionReader

        # Set up handler to use temp path for sessions
        session_manager = SessionManager(
            provider="telegram",
            chat_id="456",
            user_id="789",
            sessions_path=tmp_path,
        )
        handler._session_managers[session_manager.session_key] = session_manager

        # Handle the message
        await handler.handle_message(incoming_message)

        # Check JSONL files for stored messages
        reader = SessionReader(session_manager.session_dir)
        entries = await reader.load_entries()

        # Filter to just messages
        from ash.sessions.types import MessageEntry

        messages = [e for e in entries if isinstance(e, MessageEntry)]

        # Should have at least the user message persisted
        assert len(messages) >= 1
        assert any(m.role == "user" and m.content == "Hello!" for m in messages)
