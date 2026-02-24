"""Tests for provider implementations."""

from datetime import UTC, datetime, timedelta
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
            mock_bot.send_photo = AsyncMock()
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

    async def test_send_image_uses_send_photo(self, provider):
        image_path = "artifacts/screen.png"
        provider._bot.send_photo.return_value = MagicMock(message_id=321)
        with patch("ash.providers.telegram.provider.FSInputFile") as fs_input:
            msg_id = await provider.send(
                OutgoingMessage(
                    chat_id="123",
                    text="Here is the screenshot",
                    image_path=image_path,
                )
            )

        assert msg_id == "321"
        fs_input.assert_called_once_with(image_path)
        provider._bot.send_photo.assert_awaited_once()


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
    async def handler(self, mock_provider, mock_agent, tmp_path):
        """Create a message handler with temp sessions path."""
        handler = TelegramMessageHandler(
            provider=mock_provider,
            agent=mock_agent,
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
        handler._session_handler._session_managers[session_manager.session_key] = (
            session_manager
        )

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
        handler._session_handler._session_managers[session_manager.session_key] = (
            session_manager
        )

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
        self, mock_provider, mock_agent, incoming_message, tmp_path
    ):
        """Test handling message with non-streaming response."""
        from ash.sessions import SessionManager

        handler = TelegramMessageHandler(
            provider=mock_provider,
            agent=mock_agent,
            streaming=False,
        )

        # Set up session manager to use temp path
        session_manager = SessionManager(
            provider="telegram",
            chat_id="456",
            user_id="789",
            sessions_path=tmp_path,
        )
        handler._session_handler._session_managers[session_manager.session_key] = (
            session_manager
        )

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
        handler._session_handler._session_managers[session_manager.session_key] = (
            session_manager
        )

        session = await handler._session_handler.get_or_create_session(incoming_message)

        assert session.chat_id == "456"
        assert session.user_id == "789"
        assert session.provider == "telegram"

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
        handler._session_handler._session_managers[session_manager.session_key] = (
            session_manager
        )

        # Get session - should restore messages from JSONL
        session = await handler._session_handler.get_or_create_session(incoming_message)

        assert len(session.messages) == 2
        # Messages are in LLM format (Message objects)
        assert session.messages[0].content == "Previous message"
        assert session.messages[1].content == "Previous response"

    async def test_clear_session(self, handler, incoming_message):
        """Test clearing a session."""
        from ash.providers.telegram.handlers import SessionLock
        from ash.sessions import SessionManager

        # Set up session manager and context directly
        session_manager = SessionManager(
            provider="telegram",
            chat_id="456",
            user_id="789",
            sessions_path=handler._test_sessions_path,
        )
        session_key = session_manager.session_key
        handler._session_handler._session_managers[session_key] = session_manager
        handler._session_handler._session_contexts[session_key] = SessionLock()

        assert len(handler._session_handler._session_contexts) == 1
        assert len(handler._session_handler._session_managers) == 1

        handler.clear_session("456")
        assert len(handler._session_handler._session_contexts) == 0
        assert len(handler._session_handler._session_managers) == 0

    async def test_clear_all_sessions(self, handler, incoming_message):
        """Test clearing all sessions."""
        from ash.providers.telegram.handlers import SessionLock
        from ash.sessions import SessionManager

        # Set up session managers and contexts directly
        session_manager1 = SessionManager(
            provider="telegram",
            chat_id="456",
            user_id="789",
            sessions_path=handler._test_sessions_path,
        )
        handler._session_handler._session_managers[session_manager1.session_key] = (
            session_manager1
        )
        handler._session_handler._session_contexts[session_manager1.session_key] = (
            SessionLock()
        )

        session_manager2 = SessionManager(
            provider="telegram",
            chat_id="999",
            user_id="888",
            sessions_path=handler._test_sessions_path,
        )
        handler._session_handler._session_managers[session_manager2.session_key] = (
            session_manager2
        )
        handler._session_handler._session_contexts[session_manager2.session_key] = (
            SessionLock()
        )

        assert len(handler._session_handler._session_contexts) == 2
        assert len(handler._session_handler._session_managers) == 2

        handler.clear_all_sessions()
        assert len(handler._session_handler._session_contexts) == 0
        assert len(handler._session_handler._session_managers) == 0

    async def test_message_persistence(self, handler, incoming_message, tmp_path):
        """Test messages are persisted to JSONL files."""
        from ash.sessions import SessionReader

        # Patch sessions path so any session manager created by the handler
        # writes to tmp_path (DM threading now assigns a thread_id, creating
        # a new session manager we can't pre-register)
        with patch("ash.sessions.manager.get_sessions_path", return_value=tmp_path):
            await handler.handle_message(incoming_message)

        # Find the session directory that was created
        from ash.sessions.types import MessageEntry

        session_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(session_dirs) >= 1

        # Check at least one session has the user message
        found = False
        for session_dir in session_dirs:
            reader = SessionReader(session_dir)
            entries = await reader.load_entries()
            messages = [e for e in entries if isinstance(e, MessageEntry)]
            if any(m.role == "user" and m.content == "Hello!" for m in messages):
                found = True
                break

        assert found, "User message 'Hello!' not found in any session"

    async def test_handle_message_error_sends_error_message(
        self, handler, mock_provider, mock_agent, incoming_message
    ):
        """Test that agent failure results in error message being sent."""
        from ash.sessions import SessionManager

        # Set up session manager to use temp path
        session_manager = SessionManager(
            provider="telegram",
            chat_id="456",
            user_id="789",
            sessions_path=handler._test_sessions_path,
        )
        handler._session_handler._session_managers[session_manager.session_key] = (
            session_manager
        )

        # Create fresh async generator that raises an exception
        async def mock_stream():
            raise RuntimeError("Agent crashed!")
            yield "never reached"  # noqa: B901

        mock_agent.process_message_streaming = MagicMock(return_value=mock_stream())

        await handler.handle_message(incoming_message)

        # Verify error message was sent
        mock_provider.send.assert_called()
        call_args = mock_provider.send.call_args
        assert call_args[0][0].chat_id == "456"
        assert "error" in call_args[0][0].text.lower()

        # Verify reaction was cleared
        mock_provider.clear_reaction.assert_called_with("456", "1")

    async def test_handle_message_skips_old_messages(
        self, handler, mock_provider, mock_agent
    ):
        """Test that messages older than 5 minutes are dropped."""
        from datetime import UTC, datetime, timedelta

        from ash.providers.base import IncomingMessage
        from ash.sessions import SessionManager

        # Set up session manager to use temp path
        session_manager = SessionManager(
            provider="telegram",
            chat_id="456",
            user_id="789",
            sessions_path=handler._test_sessions_path,
        )
        handler._session_handler._session_managers[session_manager.session_key] = (
            session_manager
        )

        # Create a message that's 6 minutes old
        old_timestamp = datetime.now(UTC) - timedelta(minutes=6)
        old_message = IncomingMessage(
            id="2",
            chat_id="456",
            user_id="789",
            text="I'm old!",
            username="testuser",
            display_name="Test User",
            timestamp=old_timestamp,
        )

        await handler.handle_message(old_message)

        # Agent should NOT have been called
        mock_agent.process_message_streaming.assert_not_called()
        mock_agent.process_message.assert_not_called()

        # No response should have been sent (silent drop)
        mock_provider.send.assert_not_called()

    async def test_handle_message_skips_duplicate_messages(
        self, handler, mock_provider, mock_agent, incoming_message, tmp_path
    ):
        """Test that duplicate messages are not processed twice."""
        from ash.sessions import SessionManager

        # Set up session manager to use temp path
        session_manager = SessionManager(
            provider="telegram",
            chat_id="456",
            user_id="789",
            sessions_path=tmp_path,
        )
        handler._session_handler._session_managers[session_manager.session_key] = (
            session_manager
        )

        # Pre-seed the session with a message having the same external_id
        await session_manager.ensure_session()
        await session_manager.add_user_message(
            content="Previous message",
            token_count=10,
            metadata={"external_id": "1"},  # Same ID as incoming_message
        )

        await handler.handle_message(incoming_message)

        # Agent should NOT have been called (duplicate detected)
        mock_agent.process_message_streaming.assert_not_called()
        mock_agent.process_message.assert_not_called()

        # No response should have been sent (silent drop)
        mock_provider.send.assert_not_called()

    async def test_handle_callback_query_resumes_checkpoint(
        self, mock_provider, mock_agent, tmp_path
    ):
        """Test that inline button click resumes from checkpoint."""
        from unittest.mock import AsyncMock, MagicMock

        from ash.providers.telegram.handlers import TelegramMessageHandler

        # Create handler
        handler = TelegramMessageHandler(
            provider=mock_provider,
            agent=mock_agent,
            streaming=False,
        )

        # Set up a checkpoint in the handler
        checkpoint_id = "chkpt_test123456789"
        truncated_id = checkpoint_id[:55]
        handler._checkpoint_handler._pending_checkpoints[truncated_id] = {
            "session_key": "telegram_456_789",
            "chat_id": "456",
            "user_id": "789",
            "thread_id": None,
            "username": "testuser",
            "display_name": "Test User",
        }

        # Set up session manager
        from ash.sessions import SessionManager

        session_manager = SessionManager(
            provider="telegram",
            chat_id="456",
            user_id="789",
            sessions_path=tmp_path,
        )
        await session_manager.ensure_session()
        handler._session_handler._session_managers[session_manager.session_key] = (
            session_manager
        )

        # Store checkpoint in session log via tool_result metadata
        await session_manager.add_tool_use(
            tool_use_id="tool_123",
            name="use_agent",
            input_data={"agent": "test_agent", "message": "test"},
        )
        await session_manager.add_tool_result(
            tool_use_id="tool_123",
            output="Pausing for input",
            success=True,
            metadata={
                "checkpoint": {
                    "checkpoint_id": checkpoint_id,
                    "prompt": "Choose an option",
                    "options": ["Proceed", "Cancel"],
                }
            },
        )

        # Create mock callback query
        mock_callback = MagicMock()
        mock_callback.data = f"cp:{truncated_id}:0"  # Select option 0 ("Proceed")
        mock_callback.answer = AsyncMock()
        mock_callback.message = MagicMock()
        mock_callback.message.message_id = 100
        mock_callback.message.chat = MagicMock()
        mock_callback.message.chat.id = 456
        mock_callback.from_user = MagicMock()
        mock_callback.from_user.id = 789
        mock_callback.from_user.username = "testuser"
        mock_callback.from_user.full_name = "Test User"

        # Handle the callback - should fall back to message flow
        # (since no tool_registry/agent context)
        await handler.handle_callback_query(mock_callback)

        # Verify the callback was answered
        mock_callback.answer.assert_called()

    async def test_checkpoint_recovery_from_session_log(
        self, mock_provider, mock_agent, tmp_path
    ):
        """Test checkpoint restored after handler restart (empty in-memory cache)."""

        from ash.providers.telegram.handlers import TelegramMessageHandler
        from ash.sessions import SessionManager

        # Create session manager and store checkpoint in session log
        session_manager = SessionManager(
            provider="telegram",
            chat_id="456",
            user_id="789",
            sessions_path=tmp_path,
        )
        await session_manager.ensure_session()

        checkpoint_id = "chkpt_recovery_test_12345"
        truncated_id = checkpoint_id[:55]

        # Store checkpoint via tool_result metadata (as done in real flow)
        await session_manager.add_tool_use(
            tool_use_id="tool_456",
            name="use_agent",
            input_data={"agent": "recovery_agent", "message": "test"},
        )
        await session_manager.add_tool_result(
            tool_use_id="tool_456",
            output="Pausing for input",
            success=True,
            metadata={
                "checkpoint": {
                    "checkpoint_id": checkpoint_id,
                    "prompt": "Continue?",
                    "options": ["Yes", "No"],
                }
            },
        )

        # Create a NEW handler instance (simulating restart)
        # The in-memory checkpoint cache should be empty
        handler = TelegramMessageHandler(
            provider=mock_provider,
            agent=mock_agent,
            streaming=False,
        )

        # Verify in-memory cache is empty
        assert truncated_id not in handler._checkpoint_handler._pending_checkpoints

        # Register the session manager so disk recovery works
        handler._session_handler._session_managers[session_manager.session_key] = (
            session_manager
        )

        # Try to recover checkpoint using get_checkpoint (disk recovery path)
        routing, checkpoint = await handler._checkpoint_handler.get_checkpoint(
            truncated_id,
            response_external_id="100",  # Dummy ID
            chat_id="456",
            user_id="789",
        )

        # Should have recovered from disk
        assert checkpoint is not None
        assert checkpoint["checkpoint_id"] == checkpoint_id
        assert checkpoint["options"] == ["Yes", "No"]
        assert routing is not None
        assert routing["chat_id"] == "456"

    async def test_handle_passive_message_throttling(
        self, mock_provider, mock_agent, tmp_path, caplog
    ):
        """Test that throttled passive messages are dropped."""
        import logging
        from unittest.mock import MagicMock

        from ash.config.models import PassiveListeningConfig
        from ash.providers.base import IncomingMessage
        from ash.providers.telegram.handlers import TelegramMessageHandler
        from ash.providers.telegram.passive import PassiveEngagementThrottler

        # Set up provider attributes needed for passive listening
        mock_provider.passive_config = PassiveListeningConfig(
            enabled=True,
            chat_cooldown_minutes=30,
            max_engagements_per_hour=5,
        )
        mock_provider.bot_username = "ash_bot"

        handler = TelegramMessageHandler(
            provider=mock_provider,
            agent=mock_agent,
            streaming=False,
        )

        # Manually inject a throttler that always blocks
        mock_throttler = MagicMock(spec=PassiveEngagementThrottler)
        mock_throttler.should_consider.return_value = False
        handler._passive_handler._passive_throttler = mock_throttler  # type: ignore[union-attr]

        # Also need decider and memory_manager for the handler to proceed
        handler._passive_handler._passive_decider = MagicMock()  # type: ignore[union-attr]
        handler._passive_handler._memory_manager = MagicMock()  # type: ignore[union-attr]

        # Message that does NOT mention the bot (won't bypass throttle)
        passive_message = IncomingMessage(
            id="99",
            chat_id="group_123",
            user_id="user_456",
            text="Hey everyone, what's for lunch?",
            username="otheruser",
            display_name="Other User",
        )

        with caplog.at_level(logging.INFO, logger="telegram"):
            await handler.handle_passive_message(passive_message)

        # Agent should NOT have been called (throttled)
        mock_agent.process_message_streaming.assert_not_called()
        mock_agent.process_message.assert_not_called()

        # No response should have been sent
        mock_provider.send.assert_not_called()

        # Throttler was consulted
        mock_throttler.should_consider.assert_called_once_with("group_123")
        skipped = next(
            (r for r in caplog.records if r.msg == "passive_engagement_skipped"), None
        )
        assert skipped is not None
        assert getattr(skipped, "decision_path", None) == "throttled"
        assert getattr(skipped, "engagement_reason", None) == "rate_limiter"

    async def test_handle_passive_message_engages_on_name_mention(
        self, mock_provider, mock_agent, tmp_path, caplog
    ):
        """Test that bot name mention bypasses throttle and engages."""
        import logging
        from unittest.mock import MagicMock

        from ash.config.models import PassiveListeningConfig
        from ash.providers.base import IncomingMessage
        from ash.providers.telegram.handlers import TelegramMessageHandler
        from ash.providers.telegram.passive import PassiveEngagementThrottler
        from ash.sessions import SessionManager

        # Set up provider attributes
        mock_provider.passive_config = PassiveListeningConfig(
            enabled=True,
            chat_cooldown_minutes=30,
            max_engagements_per_hour=5,
        )
        mock_provider.bot_username = "ash_bot"
        mock_provider.edit = AsyncMock(return_value=None)
        mock_provider.delete = AsyncMock(return_value=None)

        handler = TelegramMessageHandler(
            provider=mock_provider,
            agent=mock_agent,
            streaming=True,  # Use streaming
        )

        # Inject throttler that would block (but should be bypassed)
        mock_throttler = MagicMock(spec=PassiveEngagementThrottler)
        mock_throttler.should_consider.return_value = False
        mock_throttler.record_engagement = MagicMock()
        handler._passive_handler._passive_throttler = mock_throttler  # type: ignore[union-attr]

        # Need decider and memory_manager
        handler._passive_handler._passive_decider = MagicMock()  # type: ignore[union-attr]
        handler._passive_handler._memory_manager = MagicMock()  # type: ignore[union-attr]

        # Set up session manager for the processing
        session_manager = SessionManager(
            provider="telegram",
            chat_id="group_123",
            user_id="user_456",
            sessions_path=tmp_path,
        )
        handler._session_handler._session_managers[session_manager.session_key] = (
            session_manager
        )

        # Create fresh async generator for this test
        async def mock_stream():
            yield "Response from bot"

        mock_agent.process_message_streaming = MagicMock(return_value=mock_stream())

        # Message mentions bot by name - should bypass throttle
        passive_message = IncomingMessage(
            id="99",
            chat_id="group_123",
            user_id="user_456",
            text="Hey Ash, what do you think about this?",
            username="otheruser",
            display_name="Other User",
        )

        with caplog.at_level(logging.INFO, logger="telegram"):
            await handler.handle_passive_message(passive_message)

        # Throttler.should_consider should NOT have been called (bypassed)
        mock_throttler.should_consider.assert_not_called()

        # Engagement should be recorded
        mock_throttler.record_engagement.assert_called_once_with("group_123")

        # Agent SHOULD have been called (name mention bypasses throttle)
        mock_agent.process_message_streaming.assert_called()
        engaging = next(
            (r for r in caplog.records if r.msg == "passive_engaging"), None
        )
        assert engaging is not None
        assert getattr(engaging, "decision_path", None) == "name_mentioned_fast_path"
        assert getattr(engaging, "engagement_reason", None) == "name_mentioned"

    async def test_handle_passive_message_direct_followup_bypasses_throttle_but_uses_decider(
        self, mock_provider, mock_agent, tmp_path, caplog
    ):
        """Recent post-reply follow-up should bypass throttle but still run LLM decision."""
        import logging

        from ash.chats.history import ChatHistoryWriter
        from ash.config.models import PassiveListeningConfig
        from ash.providers.base import IncomingMessage
        from ash.providers.telegram.handlers import TelegramMessageHandler
        from ash.providers.telegram.passive import PassiveEngagementThrottler

        mock_provider.passive_config = PassiveListeningConfig(
            enabled=True,
            chat_cooldown_minutes=30,
            direct_followup_window_seconds=120,
        )
        mock_provider.bot_username = "ash_bot"

        handler = TelegramMessageHandler(
            provider=mock_provider,
            agent=mock_agent,
            streaming=False,
        )

        # Create recent direct interaction history:
        # passive-engaged user message -> bot reply -> current user follow-up (no mention).
        writer = ChatHistoryWriter("telegram", "group_123")
        writer.record_user_message(
            content="@ash_bot can you help?",
            created_at=datetime.now(UTC) - timedelta(seconds=30),
            user_id="user_456",
            username="otheruser",
            metadata={"external_id": "97", "processing_mode": "passive"},
        )
        writer.record_bot_message(
            content="Sure, what's up?",
            created_at=datetime.now(UTC) - timedelta(seconds=20),
            metadata={"external_id": "98"},
        )
        writer.record_user_message(
            content="its about deploys",
            created_at=datetime.now(UTC) - timedelta(seconds=5),
            user_id="user_456",
            username="otheruser",
            metadata={"external_id": "99", "processing_mode": "passive"},
        )

        # Throttler would block if consulted, but follow-up should bypass it.
        mock_throttler = MagicMock(spec=PassiveEngagementThrottler)
        mock_throttler.should_consider.return_value = False
        handler._passive_handler._passive_throttler = mock_throttler  # type: ignore[union-attr]

        # Decision still goes through decider and returns SILENT.
        mock_decider = MagicMock()
        mock_decider.decide = AsyncMock(return_value=False)
        handler._passive_handler._passive_decider = mock_decider  # type: ignore[union-attr]
        handler._passive_handler._memory_manager = MagicMock()  # type: ignore[union-attr]

        passive_message = IncomingMessage(
            id="99",
            chat_id="group_123",
            user_id="user_456",
            text="its about deploys",
            username="otheruser",
            display_name="Other User",
            timestamp=datetime.now(UTC),
        )

        with caplog.at_level(logging.INFO, logger="telegram"):
            await handler.handle_passive_message(passive_message)

        mock_throttler.should_consider.assert_not_called()
        mock_decider.decide.assert_awaited_once()
        mock_agent.process_message.assert_not_called()
        mock_agent.process_message_streaming.assert_not_called()
        silent = next(
            (r for r in caplog.records if r.msg == "passive_engagement_silent"), None
        )
        assert silent is not None
        assert getattr(silent, "decision_path", None) == "direct_followup_llm_silent"
        assert getattr(silent, "engagement_reason", None) == "direct_followup"

    async def test_handle_passive_message_no_direct_followup_uses_throttle(
        self, mock_provider, mock_agent, tmp_path
    ):
        """Without recent direct interaction, passive flow should use throttler."""
        from ash.chats.history import ChatHistoryWriter
        from ash.config.models import PassiveListeningConfig
        from ash.providers.base import IncomingMessage
        from ash.providers.telegram.handlers import TelegramMessageHandler
        from ash.providers.telegram.passive import PassiveEngagementThrottler

        mock_provider.passive_config = PassiveListeningConfig(
            enabled=True,
            chat_cooldown_minutes=30,
            direct_followup_window_seconds=30,
        )
        mock_provider.bot_username = "ash_bot"

        handler = TelegramMessageHandler(
            provider=mock_provider,
            agent=mock_agent,
            streaming=False,
        )

        # Old assistant reply outside follow-up window.
        writer = ChatHistoryWriter("telegram", "group_123")
        writer.record_user_message(
            content="@ash_bot help",
            created_at=datetime.now(UTC) - timedelta(minutes=3),
            user_id="user_456",
            username="otheruser",
            metadata={"external_id": "80", "processing_mode": "active"},
        )
        writer.record_bot_message(
            content="How can I help?",
            created_at=datetime.now(UTC) - timedelta(minutes=2, seconds=50),
            metadata={"external_id": "81"},
        )
        writer.record_user_message(
            content="follow up",
            created_at=datetime.now(UTC),
            user_id="user_456",
            username="otheruser",
            metadata={"external_id": "82", "processing_mode": "passive"},
        )

        mock_throttler = MagicMock(spec=PassiveEngagementThrottler)
        mock_throttler.should_consider.return_value = False
        handler._passive_handler._passive_throttler = mock_throttler  # type: ignore[union-attr]
        handler._passive_handler._passive_decider = MagicMock()  # type: ignore[union-attr]
        handler._passive_handler._memory_manager = MagicMock()  # type: ignore[union-attr]

        passive_message = IncomingMessage(
            id="82",
            chat_id="group_123",
            user_id="user_456",
            text="follow up",
            username="otheruser",
            display_name="Other User",
            timestamp=datetime.now(UTC),
        )

        await handler.handle_passive_message(passive_message)

        mock_throttler.should_consider.assert_called_once_with("group_123")

    async def test_passive_extraction_includes_recent_same_user_context(
        self, mock_provider, mock_agent
    ):
        """Passive extraction should include recent same-speaker user context."""
        from ash.chats.history import ChatHistoryWriter
        from ash.llm.types import Role
        from ash.providers.base import IncomingMessage
        from ash.providers.telegram.handlers import TelegramMessageHandler

        handler = TelegramMessageHandler(
            provider=mock_provider,
            agent=mock_agent,
            streaming=False,
        )

        mock_extractor = MagicMock()
        mock_extractor.extract_from_message = AsyncMock(return_value=0)
        handler._passive_handler._passive_extractor = mock_extractor  # type: ignore[union-attr]

        writer = ChatHistoryWriter("telegram", "group_123")
        writer.record_user_message(
            content="Randolf is going to Tokyo in May",
            user_id="user_456",
            username="randolf",
            display_name="Randolf",
            metadata={"external_id": "100"},
        )
        writer.record_user_message(
            content="I think the flights are expensive",
            user_id="user_456",
            username="randolf",
            display_name="Randolf",
            metadata={"external_id": "101"},
        )

        passive_message = IncomingMessage(
            id="102",
            chat_id="group_123",
            user_id="user_456",
            text="he's still going in May",
            username="randolf",
            display_name="Randolf",
        )

        await handler._passive_handler._extract_passive_memories(passive_message)  # type: ignore[union-attr]

        mock_extractor.extract_from_message.assert_awaited_once()
        call = mock_extractor.extract_from_message.call_args.kwargs
        recent_messages = call["recent_user_messages"]
        assert len(recent_messages) >= 2
        assert any("Tokyo in May" in m.content for m in recent_messages)
        assert recent_messages[-1].role == Role.USER
        assert "he's still going in May" in recent_messages[-1].content
