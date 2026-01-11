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
            return_value=MagicMock(text="Response from agent")
        )

        async def mock_stream():
            yield "Response "
            yield "from "
            yield "agent"

        agent.process_message_streaming = MagicMock(return_value=mock_stream())
        return agent

    @pytest.fixture
    async def handler(self, mock_provider, mock_agent, database):
        """Create a message handler."""
        return TelegramMessageHandler(
            provider=mock_provider,
            agent=mock_agent,
            database=database,
            streaming=True,
        )

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
        self, handler, mock_provider, incoming_message
    ):
        """Test that handling a message sends typing indicator."""
        await handler.handle_message(incoming_message)

        mock_provider.send_typing.assert_called_once_with("456")

    async def test_handle_message_streaming(
        self, handler, mock_provider, incoming_message
    ):
        """Test handling message with streaming response."""
        await handler.handle_message(incoming_message)

        mock_provider.send_streaming.assert_called_once()
        call_kwargs = mock_provider.send_streaming.call_args.kwargs
        assert call_kwargs["chat_id"] == "456"
        assert call_kwargs["reply_to"] == "1"

    async def test_handle_message_non_streaming(
        self, mock_provider, mock_agent, database, incoming_message
    ):
        """Test handling message with non-streaming response."""
        handler = TelegramMessageHandler(
            provider=mock_provider,
            agent=mock_agent,
            database=database,
            streaming=False,
        )

        await handler.handle_message(incoming_message)

        mock_agent.process_message.assert_called_once()
        mock_provider.send.assert_called_once()

    async def test_session_creation(self, handler, incoming_message):
        """Test session is created for new chat."""
        session = await handler._get_or_create_session(incoming_message)

        assert session.chat_id == "456"
        assert session.user_id == "789"
        assert session.provider == "telegram"

    async def test_session_reuse(self, handler, incoming_message):
        """Test session is reused for same chat."""
        session1 = await handler._get_or_create_session(incoming_message)
        session2 = await handler._get_or_create_session(incoming_message)

        assert session1 is session2

    async def test_session_restoration(self, handler, database, incoming_message):
        """Test messages are restored from database."""
        from ash.memory.store import MemoryStore

        # Pre-populate database with messages
        async with database.session() as db_session:
            store = MemoryStore(db_session)
            db_sess = await store.get_or_create_session(
                provider="telegram",
                chat_id="456",
                user_id="789",
            )
            await store.add_message(
                session_id=db_sess.id,
                role="user",
                content="Previous message",
            )
            await store.add_message(
                session_id=db_sess.id,
                role="assistant",
                content="Previous response",
            )

        # Get session - should restore messages
        session = await handler._get_or_create_session(incoming_message)

        assert len(session.messages) == 2
        assert session.messages[0].content == "Previous message"
        assert session.messages[1].content == "Previous response"

    async def test_clear_session(self, handler, incoming_message):
        """Test clearing a session."""
        await handler._get_or_create_session(incoming_message)
        assert len(handler._sessions) == 1

        handler.clear_session("456")
        assert len(handler._sessions) == 0

    async def test_clear_all_sessions(self, handler, incoming_message):
        """Test clearing all sessions."""
        await handler._get_or_create_session(incoming_message)

        # Create another session
        msg2 = IncomingMessage(
            id="2",
            chat_id="999",
            user_id="888",
            text="Hi",
        )
        await handler._get_or_create_session(msg2)
        assert len(handler._sessions) == 2

        handler.clear_all_sessions()
        assert len(handler._sessions) == 0

    async def test_message_persistence(self, handler, database, incoming_message):
        """Test messages are persisted to database."""
        await handler.handle_message(incoming_message)

        # Check database for stored message
        from ash.memory.store import MemoryStore

        async with database.session() as db_session:
            store = MemoryStore(db_session)
            # Get the session we just used
            session = await store.get_or_create_session(
                provider="telegram",
                chat_id="456",
                user_id="789",
            )
            # Get messages for this session
            messages = await store.get_messages(session.id)
            # Should have at least the user message persisted
            assert len(messages) >= 1
            assert any(m.role == "user" and m.content == "Hello!" for m in messages)
