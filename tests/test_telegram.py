"""Tests for Telegram provider spec behaviors.

Tests key behaviors from specs/telegram.md:
- Message conversion (Telegram -> IncomingMessage)
- Thinking message formatting
- User/group authorization
- Mention detection and stripping
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.enums import ParseMode

from ash.providers.base import OutgoingMessage
from ash.providers.telegram.formatting import rendered_text_length
from ash.providers.telegram.handlers import (
    ProgressMessageTool,
    ToolTracker,
    escape_markdown_v2,
    format_tool_brief,
)
from ash.providers.telegram.handlers.provenance import ProvenanceState
from ash.providers.telegram.handlers.utils import (
    append_inline_attribution,
    merge_progress_and_response,
)
from ash.providers.telegram.provider import TelegramProvider
from ash.tools.base import ToolContext, ToolResult


class TestMessageConversion:
    """Test Telegram message to IncomingMessage conversion."""

    @pytest.fixture
    def provider(self):
        """Create a Telegram provider with mock bot."""
        with patch("ash.providers.telegram.provider.Bot"):
            provider = TelegramProvider(
                bot_token="test_token",
                allowed_users=["@testuser"],
            )
            provider._bot_username = "testbot"
            yield provider

    def test_to_incoming_message_basic(self, provider):
        """Test basic message conversion preserves all fields."""
        mock_message = MagicMock()
        mock_message.message_id = 123
        mock_message.chat.id = 456
        mock_message.chat.type = "private"
        mock_message.chat.title = None
        mock_message.from_user.full_name = "Test User"
        mock_message.reply_to_message = None
        mock_message.message_thread_id = None

        incoming = provider._to_incoming_message(
            mock_message,
            user_id=789,
            username="testuser",
            text="Hello, bot!",
        )

        assert incoming.id == "123"
        assert incoming.chat_id == "456"
        assert incoming.user_id == "789"
        assert incoming.text == "Hello, bot!"
        assert incoming.username == "testuser"
        assert incoming.display_name == "Test User"
        assert incoming.metadata["chat_type"] == "private"
        assert incoming.images == []

    def test_to_incoming_message_with_reply(self, provider):
        """Test message conversion includes reply_to_message_id."""
        mock_message = MagicMock()
        mock_message.message_id = 123
        mock_message.chat.id = 456
        mock_message.chat.type = "private"
        mock_message.chat.title = None
        mock_message.from_user.full_name = "Test User"
        mock_message.reply_to_message = MagicMock()
        mock_message.reply_to_message.message_id = 100
        mock_message.message_thread_id = None

        incoming = provider._to_incoming_message(
            mock_message,
            user_id=789,
            username="testuser",
            text="Replying to you",
        )

        assert incoming.reply_to_message_id == "100"

    def test_to_incoming_message_group_with_thread(self, provider):
        """Test group message conversion includes thread_id for forum topics."""
        mock_message = MagicMock()
        mock_message.message_id = 123
        mock_message.chat.id = -1001234567890
        mock_message.chat.type = "supergroup"
        mock_message.chat.title = "Test Group"
        mock_message.from_user.full_name = "Test User"
        mock_message.reply_to_message = None
        mock_message.message_thread_id = 42  # Forum topic thread

        incoming = provider._to_incoming_message(
            mock_message,
            user_id=789,
            username="testuser",
            text="Message in forum topic",
        )

        assert incoming.metadata["chat_type"] == "supergroup"
        assert incoming.metadata["chat_title"] == "Test Group"
        assert incoming.metadata["thread_id"] == "42"


class TestToolBriefFormatting:
    """Test tool brief formatting for thinking messages."""

    def test_bash_command(self):
        """Test bash command formatting."""
        brief = format_tool_brief("bash", {"command": "git status"})
        assert brief == "Running: `git status`"

    def test_bash_command_truncation(self):
        """Test long bash commands are truncated."""
        long_cmd = "echo " + "x" * 100
        brief = format_tool_brief("bash", {"command": long_cmd})
        assert len(brief) < len(long_cmd) + 20
        assert "..." in brief

    def test_web_search(self):
        """Test web search formatting."""
        brief = format_tool_brief("web_search", {"query": "python async"})
        assert brief == "Searching: python async"

    def test_web_fetch(self):
        """Test web fetch shows domain only."""
        brief = format_tool_brief(
            "web_fetch", {"url": "https://docs.python.org/3/library/asyncio.html"}
        )
        assert brief == "Reading: docs.python.org"

    def test_read_file(self):
        """Test read file shows filename only."""
        brief = format_tool_brief(
            "read_file", {"file_path": "/home/user/project/config.py"}
        )
        assert brief == "Reading: config.py"

    def test_write_file(self):
        """Test write file shows filename only."""
        brief = format_tool_brief("write_file", {"file_path": "/home/user/output.txt"})
        assert brief == "Writing: output.txt"

    def test_remember(self):
        """Test remember tool formatting."""
        brief = format_tool_brief("remember", {"content": "User prefers dark mode"})
        assert brief == "Saving to memory"

    def test_recall(self):
        """Test recall tool formatting."""
        brief = format_tool_brief("recall", {"query": "user preferences"})
        assert brief == "Searching memories: user preferences"

    def test_unknown_tool(self):
        """Test unknown tool uses generic format (underscores removed, _tool suffix stripped)."""
        brief = format_tool_brief("custom_tool", {"param": "value"})
        # Implementation strips _tool suffix and replaces underscores with spaces
        assert brief == "Running: custom"


class TestMarkdownEscaping:
    """Test MarkdownV2 escaping for Telegram."""

    def test_escape_periods(self):
        """Test periods are escaped for MarkdownV2."""
        result = escape_markdown_v2("Thinking...")
        assert result == "Thinking\\.\\.\\."

    def test_escape_parentheses(self):
        """Test parentheses are escaped."""
        result = escape_markdown_v2("(test)")
        assert result == "\\(test\\)"

    def test_escape_special_chars(self):
        """Test all special chars are escaped."""
        # Special chars: _ * [ ] ( ) ~ ` > # + - = | { } . !
        result = escape_markdown_v2("_*[]()~`>#+-=|{}.!")
        # Each char should be preceded by backslash
        assert result == "\\_\\*\\[\\]\\(\\)\\~\\`\\>\\#\\+\\-\\=\\|\\{\\}\\.\\!"

    def test_escape_normal_text_unchanged(self):
        """Test normal text passes through unchanged."""
        result = escape_markdown_v2("Hello world")
        assert result == "Hello world"


class TestUserAuthorization:
    """Test user authorization logic."""

    def test_user_allowed_by_username(self):
        """Test user allowed by @username."""
        with patch("ash.providers.telegram.provider.Bot"):
            provider = TelegramProvider(
                bot_token="test",
                allowed_users=["@alice", "@bob"],
            )
            assert provider._is_user_allowed(0, "alice") is True
            assert provider._is_user_allowed(0, "charlie") is False

    def test_user_allowed_by_id(self):
        """Test user allowed by numeric ID."""
        with patch("ash.providers.telegram.provider.Bot"):
            provider = TelegramProvider(
                bot_token="test",
                allowed_users=["12345", "67890"],
            )
            assert provider._is_user_allowed(12345, None) is True
            assert provider._is_user_allowed(99999, None) is False

    def test_empty_allowed_users_permits_all(self):
        """Test empty allowed_users list permits all users."""
        with patch("ash.providers.telegram.provider.Bot"):
            provider = TelegramProvider(
                bot_token="test",
                allowed_users=[],
            )
            assert provider._is_user_allowed(12345, "anyone") is True


class TestMentionDetection:
    """Test bot mention detection and stripping for groups."""

    @pytest.fixture
    def provider(self):
        """Create provider with bot username set."""
        with patch("ash.providers.telegram.provider.Bot"):
            provider = TelegramProvider(bot_token="test")
            provider._bot_username = "testbot"
            yield provider

    def test_is_mentioned_in_text(self, provider):
        """Test mention detected in message text."""
        mock_message = MagicMock()
        mock_message.text = "Hey @testbot what's up?"
        mock_message.caption = None
        mock_message.entities = []
        mock_message.caption_entities = []

        assert provider._is_mentioned(mock_message) is True

    def test_is_mentioned_case_insensitive(self, provider):
        """Test mention detection is case-insensitive."""
        mock_message = MagicMock()
        mock_message.text = "Hey @TestBot what's up?"
        mock_message.caption = None
        mock_message.entities = []
        mock_message.caption_entities = []

        assert provider._is_mentioned(mock_message) is True

    def test_is_not_mentioned(self, provider):
        """Test no mention detected when bot not mentioned."""
        mock_message = MagicMock()
        mock_message.text = "Hey @otherbot what's up?"
        mock_message.caption = None
        mock_message.entities = []
        mock_message.caption_entities = []

        assert provider._is_mentioned(mock_message) is False

    def test_strip_mention(self, provider):
        """Test bot mention is stripped from text."""
        result = provider._strip_mention("@testbot hello there")
        assert result == "hello there"

    def test_strip_mention_middle_of_text(self, provider):
        """Test mention stripped from middle of text (leaves extra space)."""
        result = provider._strip_mention("hey @testbot can you help?")
        # Implementation uses regex substitution which may leave extra space
        assert result == "hey  can you help?"

    def test_strip_mention_case_insensitive(self, provider):
        """Test mention stripping is case-insensitive."""
        result = provider._strip_mention("@TestBot hello")
        assert result == "hello"

    def test_strip_mention_preserves_other_mentions(self, provider):
        """Test other mentions are preserved."""
        result = provider._strip_mention("@testbot tell @alice hello")
        assert result == "tell @alice hello"


class TestReplyDetection:
    """Test _is_reply() targeting — only replies to bot's own messages count."""

    @pytest.fixture
    def provider(self):
        """Create provider with bot ID set."""
        with patch("ash.providers.telegram.provider.Bot"):
            provider = TelegramProvider(bot_token="test")
            provider._bot_username = "testbot"
            provider._bot_id = 999
            yield provider

    def _make_message(self, reply_from_id: int | None = None) -> MagicMock:
        """Create a mock TelegramMessage with optional reply target."""
        msg = MagicMock()
        if reply_from_id is not None:
            msg.reply_to_message = MagicMock()
            msg.reply_to_message.from_user = MagicMock()
            msg.reply_to_message.from_user.id = reply_from_id
        else:
            msg.reply_to_message = None
        return msg

    def test_reply_to_bot_message(self, provider):
        """Reply to the bot's own message → True (active mode)."""
        msg = self._make_message(reply_from_id=999)
        assert provider._is_reply(msg) is True

    def test_reply_to_other_user(self, provider):
        """Reply to another user's message → False (passive or skip)."""
        msg = self._make_message(reply_from_id=123)
        assert provider._is_reply(msg) is False

    def test_no_reply(self, provider):
        """Not a reply at all → False."""
        msg = self._make_message(reply_from_id=None)
        assert provider._is_reply(msg) is False

    def test_bot_id_none_falls_back(self, provider):
        """When _bot_id is None, fall back to old behavior (any reply → True)."""
        provider._bot_id = None
        msg = self._make_message(reply_from_id=123)
        assert provider._is_reply(msg) is True

    def test_reply_to_message_without_from_user(self, provider):
        """Reply to a message with no from_user (e.g. channel post) → False."""
        msg = MagicMock()
        msg.reply_to_message = MagicMock()
        msg.reply_to_message.from_user = None
        assert provider._is_reply(msg) is False


class TestGroupProcessingMode:
    """Test that reply targeting integrates correctly with _should_process_message."""

    @pytest.fixture
    def provider(self):
        """Create provider configured for group mention mode."""
        with patch("ash.providers.telegram.provider.Bot"):
            provider = TelegramProvider(
                bot_token="test",
                allowed_groups=["-100"],
                group_mode="mention",
            )
            provider._bot_username = "testbot"
            provider._bot_id = 999
            yield provider

    def _make_group_message(
        self,
        *,
        text: str = "hello",
        reply_from_id: int | None = None,
        user_id: int = 1,
        username: str = "alice",
    ) -> MagicMock:
        msg = MagicMock()
        msg.chat.id = -100
        msg.chat.type = "group"
        msg.chat.title = "Test Group"
        msg.message_id = 42
        msg.from_user.id = user_id
        msg.from_user.username = username
        msg.from_user.full_name = username.title()
        msg.text = text
        msg.caption = None
        msg.entities = []
        msg.caption_entities = []
        msg.date = None
        msg.message_thread_id = None
        if reply_from_id is not None:
            msg.reply_to_message = MagicMock()
            msg.reply_to_message.from_user = MagicMock()
            msg.reply_to_message.from_user.id = reply_from_id
        else:
            msg.reply_to_message = None
        return msg

    @pytest.mark.asyncio
    async def test_reply_to_bot_is_active(self, provider):
        """Reply to bot's message in group → active processing."""
        with patch("ash.chats.ChatHistoryWriter"):
            msg = self._make_group_message(reply_from_id=999)
            result = await provider._should_process_message(msg)
        assert result is not None
        assert result[0] == "active"

    @pytest.mark.asyncio
    async def test_reply_to_other_user_is_skipped(self, provider):
        """Reply to another user in group → skipped (no passive handler)."""
        with patch("ash.chats.ChatHistoryWriter"):
            msg = self._make_group_message(reply_from_id=123)
            result = await provider._should_process_message(msg)
        assert result is None

    @pytest.mark.asyncio
    async def test_mention_plus_reply_to_other_is_active(self, provider):
        """@mention + reply to another user → active (mention takes priority)."""
        with patch("ash.chats.ChatHistoryWriter"):
            msg = self._make_group_message(
                text="@testbot what do you think?", reply_from_id=123
            )
            result = await provider._should_process_message(msg)
        assert result is not None
        assert result[0] == "active"


class TestToolTracker:
    """Test ToolTracker for thinking message management."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider for tracker tests."""
        provider = MagicMock()
        provider.send = AsyncMock(return_value="msg_123")
        provider.edit = AsyncMock()
        return provider

    @pytest.fixture
    def tracker(self, mock_provider):
        """Create a ToolTracker instance."""
        return ToolTracker(
            provider=mock_provider,
            chat_id="456",
            reply_to="789",
        )

    async def test_first_tool_creates_thinking_message(self, tracker, mock_provider):
        """Test first tool call creates the thinking message."""
        await tracker.on_tool_start("bash", {"command": "ls"})

        assert tracker.thinking_msg_id == "msg_123"
        assert tracker.tool_count == 1
        mock_provider.send.assert_called_once()
        # Check that the message contains thinking status
        call_args = mock_provider.send.call_args
        msg = call_args[0][0]
        assert isinstance(msg, OutgoingMessage)
        assert "Thinking" in msg.text

    async def test_subsequent_tools_edit_message(self, tracker, mock_provider):
        """Test subsequent tool calls edit the thinking message."""
        await tracker.on_tool_start("bash", {"command": "ls"})
        await tracker.on_tool_start("bash", {"command": "pwd"})

        assert tracker.tool_count == 2
        mock_provider.edit.assert_called_once()

    async def test_finalize_response_edits_thinking_message(
        self, tracker, mock_provider
    ):
        """Test finalize_response edits thinking message with final content."""
        await tracker.on_tool_start("bash", {"command": "ls"})
        msg_id = await tracker.finalize_response("Here's the result")

        assert msg_id == "msg_123"
        mock_provider.edit.assert_called()
        # Final edit should include the response
        final_call = mock_provider.edit.call_args
        assert "Here's the result" in final_call[0][2]

    async def test_finalize_response_no_tools_sends_directly(
        self, tracker, mock_provider
    ):
        """Test finalize_response sends new message when no tools were used."""
        msg_id = await tracker.finalize_response("Quick response")

        # Should send, not edit (no thinking message exists)
        assert msg_id == "msg_123"
        mock_provider.send.assert_called_once()

    async def test_finalize_response_with_progress_messages(
        self, tracker, mock_provider
    ):
        """Test finalize_response includes progress messages in final content."""
        await tracker.on_tool_start("bash", {"command": "ls"})
        tracker.add_progress_message("Step 1 done")
        msg_id = await tracker.finalize_response("Here's the result")

        assert msg_id == "msg_123"
        final_call = mock_provider.edit.call_args
        final_text = final_call[0][2]
        assert "Step 1 done" in final_text
        assert "Here's the result" in final_text
        # No stats or thinking text in final output
        assert "Thinking" not in final_text
        assert "tool call" not in final_text

    async def test_progress_tool_image_passthrough_sends_direct(
        self, tracker, mock_provider
    ):
        tool = ProgressMessageTool(tracker)
        context = ToolContext(
            session_id="session-1",
            user_id="user-1",
            chat_id="456",
            provider="telegram",
            metadata={"reply_to_message_id": "789"},
        )
        result = await tool.execute(
            {
                "message": "screenshot attached",
                "image_path": "/workspace/screenshot.png",
            },
            context,
        )

        assert result.is_error is False
        assert result.metadata is not None
        assert result.metadata.get("sent_message_id") == "msg_123"
        mock_provider.send.assert_called()
        sent = mock_provider.send.call_args[0][0]
        assert sent.image_path == "/workspace/screenshot.png"
        assert sent.text == "screenshot attached"

    def test_display_truncation_uses_rendered_markdown_v2_length(self, tracker):
        tracker.progress_messages = ["." * 3000, "." * 3000]
        display = tracker._build_display_message()

        assert rendered_text_length(display, ParseMode.MARKDOWN_V2) <= 4096


class TestProgressResponseMerge:
    def test_merges_progress_and_response(self):
        merged = merge_progress_and_response(["Step 1"], "Done")
        assert merged == "Step 1\n\nDone"

    def test_dedupes_identical_trailing_response(self):
        merged = merge_progress_and_response(["Done"], "Done")
        assert merged == "Done"


class TestResponseAttribution:
    def test_appends_inline_attribution(self):
        text = append_inline_attribution(
            "Here is the answer.", "I checked docs.python.org."
        )
        assert text == "Here is the answer. I checked docs.python.org."

    def test_skips_when_no_attribution(self):
        text = append_inline_attribution("Plain response", None)
        assert text == "Plain response"


class TestProvenanceState:
    def test_collects_top_domains_and_skills(self):
        state = ProvenanceState()
        state.add_from_tool(
            "web_search",
            {"query": "python"},
            ToolResult.success("ok", domains=["docs.python.org", "docs.python.org"]),
        )
        state.add_from_tool(
            "web_fetch",
            {"url": "https://example.com/page"},
            ToolResult.success("ok", final_url="https://example.com/page"),
        )
        state.add_from_tool(
            "use_skill",
            {"skill": "dex", "message": "track task"},
            ToolResult.success("ok"),
        )
        state.add_from_tool(
            "web_fetch",
            {"url": "https://one.test"},
            ToolResult.success("ok", final_url="https://one.test"),
        )
        state.add_from_tool(
            "web_fetch",
            {"url": "https://two.test"},
            ToolResult.success("ok", final_url="https://two.test"),
        )

        attribution = state.render_inline(max_domains=3)
        assert attribution is not None
        assert "docs.python.org" in attribution
        assert "example.com" in attribution
        assert "one.test" in attribution
        assert "two.test" not in attribution
        assert "/dex" in attribution

    def test_ignores_tool_errors(self):
        state = ProvenanceState()
        state.add_from_tool(
            "web_fetch",
            {"url": "https://ignored.test"},
            ToolResult.error("failed", final_url="https://ignored.test"),
        )
        assert state.render_inline() is None


class TestTrackerProvenance:
    async def test_tracker_builds_provenance_clause(self):
        provider = MagicMock()
        tracker = ToolTracker(provider=provider, chat_id="456", reply_to="789")
        await tracker.on_tool_complete(
            "web_fetch",
            {"url": "https://docs.python.org/3/"},
            ToolResult.success("ok", final_url="https://docs.python.org/3/"),
        )
        await tracker.on_tool_complete(
            "use_skill",
            {"skill": "dex", "message": "update task"},
            ToolResult.success("ok"),
        )

        clause = tracker.build_provenance_clause()
        assert clause is not None
        assert "docs.python.org" in clause
        assert "/dex" in clause
