"""Tests for chat state management."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ash.chats import ChatInfo, ChatState, ChatStateManager, Participant
from ash.providers.base import IncomingMessage

# =============================================================================
# Telegram Message Fixtures
# =============================================================================


@pytest.fixture
def telegram_private_message() -> IncomingMessage:
    """A private message from a Telegram user."""
    return IncomingMessage(
        id="12345",
        chat_id="67890",
        user_id="11111",
        text="Hello there!",
        username="alice",
        display_name="Alice Smith",
        timestamp=datetime.now(UTC),
        metadata={
            "chat_type": "private",
        },
    )


@pytest.fixture
def telegram_group_message() -> IncomingMessage:
    """A message in a Telegram group."""
    return IncomingMessage(
        id="12346",
        chat_id="-100123456789",
        user_id="22222",
        text="Hey everyone!",
        username="bob",
        display_name="Bob Jones",
        timestamp=datetime.now(UTC),
        metadata={
            "chat_type": "supergroup",
            "chat_title": "My Test Group",
        },
    )


@pytest.fixture
def telegram_thread_message() -> IncomingMessage:
    """A message in a Telegram topic/thread."""
    return IncomingMessage(
        id="12347",
        chat_id="-100123456789",
        user_id="33333",
        text="Thread reply",
        username="charlie",
        display_name="Charlie Brown",
        timestamp=datetime.now(UTC),
        metadata={
            "chat_type": "supergroup",
            "chat_title": "My Test Group",
            "thread_id": "999",
        },
    )


@pytest.fixture
def telegram_message_no_username() -> IncomingMessage:
    """A Telegram message from a user without a username."""
    return IncomingMessage(
        id="12348",
        chat_id="-100123456789",
        user_id="44444",
        text="Hello from user without username",
        username=None,
        display_name="No Username User",
        timestamp=datetime.now(UTC),
        metadata={
            "chat_type": "supergroup",
            "chat_title": "My Test Group",
        },
    )


@pytest.fixture
def ash_home(tmp_path: Path) -> Path:
    """Create a temporary ash home directory."""
    ash_home = tmp_path / ".ash"
    ash_home.mkdir()
    return ash_home


@pytest.fixture
def mock_get_chat_dir(ash_home: Path):
    """Factory that creates a mock get_chat_dir function using tmp_path."""

    def _get_chat_dir(
        provider: str, chat_id: str, thread_id: str | None = None
    ) -> Path:
        base = ash_home / "chats" / provider / chat_id
        if thread_id:
            return base / "threads" / thread_id
        return base

    return _get_chat_dir


# =============================================================================
# ChatState Model Tests
# =============================================================================


class TestChatStateModel:
    """Tests for ChatState model."""

    def test_create_default_state(self):
        """Test creating a default chat state."""
        state = ChatState(chat=ChatInfo(id="123"))

        assert state.chat.id == "123"
        assert state.participants == []
        assert state.updated_at is not None

    def test_create_state_with_chat_metadata(self):
        """Test creating state with full chat info."""
        state = ChatState(
            chat=ChatInfo(
                id="-100123456789",
                type="supergroup",
                title="My Test Group",
            )
        )

        assert state.chat.id == "-100123456789"
        assert state.chat.type == "supergroup"
        assert state.chat.title == "My Test Group"

    def test_get_participant_not_found(self):
        """Test getting non-existent participant returns None."""
        state = ChatState(chat=ChatInfo(id="123"))

        assert state.get_participant("user-1") is None

    def test_get_participant_found(self):
        """Test getting existing participant."""
        state = ChatState(
            chat=ChatInfo(id="123"),
            participants=[
                Participant(
                    id="user-1",
                    username="alice",
                    display_name="Alice",
                    message_count=5,
                )
            ],
        )

        participant = state.get_participant("user-1")
        assert participant is not None
        assert participant.username == "alice"

    def test_update_participant_creates_new(self):
        """Test update_participant creates new participant if not exists."""
        state = ChatState(chat=ChatInfo(id="123"))

        participant = state.update_participant(
            user_id="user-1",
            username="alice",
            display_name="Alice",
        )

        assert participant.id == "user-1"
        assert participant.username == "alice"
        assert participant.display_name == "Alice"
        assert participant.message_count == 1
        assert participant.first_seen is not None
        assert participant.last_active is not None
        assert len(state.participants) == 1

    def test_update_participant_updates_existing(self):
        """Test update_participant updates existing participant."""
        now = datetime.now(UTC)
        state = ChatState(
            chat=ChatInfo(id="123"),
            participants=[
                Participant(
                    id="user-1",
                    username="alice",
                    display_name="Alice",
                    message_count=5,
                    first_seen=now,
                    last_active=now,
                )
            ],
        )

        participant = state.update_participant(
            user_id="user-1",
            username="alice_new",
            display_name="Alice Smith",
        )

        assert participant.username == "alice_new"
        assert participant.display_name == "Alice Smith"
        assert participant.message_count == 6
        assert participant.first_seen == now  # Should not change
        assert participant.last_active >= now  # Should be updated
        assert len(state.participants) == 1

    def test_update_participant_preserves_none_fields(self):
        """Test that None values don't overwrite existing data."""
        state = ChatState(
            chat=ChatInfo(id="123"),
            participants=[
                Participant(
                    id="user-1",
                    username="alice",
                    display_name="Alice",
                    message_count=5,
                )
            ],
        )

        # Update with None username should preserve existing
        participant = state.update_participant(
            user_id="user-1",
            username=None,
            display_name="Alice Updated",
        )

        assert participant.username == "alice"  # Preserved
        assert participant.display_name == "Alice Updated"  # Updated

    def test_multiple_participants(self):
        """Test tracking multiple participants."""
        state = ChatState(chat=ChatInfo(id="group-123"))

        state.update_participant("user-1", "alice", "Alice")
        state.update_participant("user-2", "bob", "Bob")
        state.update_participant("user-3", None, "Charlie")

        assert len(state.participants) == 3
        assert state.get_participant("user-1").username == "alice"
        assert state.get_participant("user-2").username == "bob"
        assert state.get_participant("user-3").username is None


# =============================================================================
# ChatStateManager Tests
# =============================================================================


class TestChatStateManager:
    """Tests for ChatStateManager."""

    def test_creates_default_state(self, monkeypatch, mock_get_chat_dir):
        """Test manager creates default state for new chat."""
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        manager = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
        )
        state = manager.load()

        assert state.chat.id == "-123456"
        assert state.participants == []

    def test_saves_and_loads_state(self, monkeypatch, mock_get_chat_dir):
        """Test state persistence."""
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        manager = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
        )
        manager.update_participant("user-1", "alice", "Alice")
        manager.update_chat_info(chat_type="supergroup", title="Test Group")

        # Create new manager instance to test loading
        manager2 = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
        )
        state = manager2.load()

        assert state.chat.type == "supergroup"
        assert state.chat.title == "Test Group"
        assert len(state.participants) == 1
        assert state.participants[0].id == "user-1"
        assert state.participants[0].username == "alice"

    def test_state_file_location(self, monkeypatch, mock_get_chat_dir, ash_home: Path):
        """Test that state file is written to correct location."""
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        manager = ChatStateManager(
            provider="telegram",
            chat_id="-100123456789",
        )
        manager.update_participant("user-1", "alice", "Alice")

        expected_path = ash_home / "chats" / "telegram" / "-100123456789" / "state.json"
        assert expected_path.exists()

        # Verify JSON structure
        data = json.loads(expected_path.read_text())
        assert data["chat"]["id"] == "-100123456789"
        assert len(data["participants"]) == 1

    def test_thread_creates_separate_state(self, monkeypatch, mock_get_chat_dir):
        """Test thread has separate state from parent chat."""
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        # Update chat state
        chat_manager = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
        )
        chat_manager.update_participant("user-1", "alice", "Alice")

        # Update thread state
        thread_manager = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
            thread_id="789",
        )
        thread_manager.update_participant("user-2", "bob", "Bob")

        # Verify separate states
        chat_state = chat_manager.load()
        thread_state = thread_manager.load()

        assert len(chat_state.participants) == 1
        assert chat_state.participants[0].id == "user-1"

        assert len(thread_state.participants) == 1
        assert thread_state.participants[0].id == "user-2"

    def test_thread_state_file_location(
        self, monkeypatch, mock_get_chat_dir, ash_home: Path
    ):
        """Test that thread state file is in correct nested location."""
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        manager = ChatStateManager(
            provider="telegram",
            chat_id="-100123456789",
            thread_id="999",
        )
        manager.update_participant("user-1", "alice", "Alice")

        expected_path = (
            ash_home
            / "chats"
            / "telegram"
            / "-100123456789"
            / "threads"
            / "999"
            / "state.json"
        )
        assert expected_path.exists()

    def test_update_chat_info_partial(self, monkeypatch, mock_get_chat_dir):
        """Test updating chat info partially."""
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        manager = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
        )

        # Update only type
        manager.update_chat_info(chat_type="supergroup")
        state = manager.load()
        assert state.chat.type == "supergroup"
        assert state.chat.title is None

        # Update only title
        manager.update_chat_info(title="My Group")
        state = manager.load()
        assert state.chat.type == "supergroup"  # Preserved
        assert state.chat.title == "My Group"

    def test_multiple_updates_same_participant(self, monkeypatch, mock_get_chat_dir):
        """Test multiple updates from same participant increment count."""
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        manager = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
        )

        # Simulate multiple messages from same user
        for _ in range(5):
            manager.update_participant("user-1", "alice", "Alice")

        state = manager.load()
        assert len(state.participants) == 1
        assert state.participants[0].message_count == 5


# =============================================================================
# Integration Tests - Telegram Handler
# =============================================================================


class TestTelegramChatStateIntegration:
    """Integration tests for chat state updates from Telegram handler."""

    def test_private_message_updates_state(
        self,
        monkeypatch,
        mock_get_chat_dir,
        ash_home: Path,
        telegram_private_message: IncomingMessage,
    ):
        """Test that a private message creates correct chat state."""
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        from ash.providers.telegram.handlers import TelegramMessageHandler

        # Call the _update_chat_state method directly
        handler = MagicMock(spec=TelegramMessageHandler)
        handler._provider = MagicMock()
        handler._provider.name = "telegram"

        # Import and call the actual method
        TelegramMessageHandler._update_chat_state(
            handler, telegram_private_message, thread_id=None
        )

        # Verify state file
        state_path = (
            ash_home
            / "chats"
            / "telegram"
            / telegram_private_message.chat_id
            / "state.json"
        )
        assert state_path.exists()

        data = json.loads(state_path.read_text())
        assert data["chat"]["id"] == telegram_private_message.chat_id
        assert data["chat"]["type"] == "private"
        assert len(data["participants"]) == 1
        assert data["participants"][0]["id"] == telegram_private_message.user_id
        assert data["participants"][0]["username"] == "alice"
        assert data["participants"][0]["display_name"] == "Alice Smith"

    def test_group_message_updates_state(
        self,
        monkeypatch,
        mock_get_chat_dir,
        ash_home: Path,
        telegram_group_message: IncomingMessage,
    ):
        """Test that a group message creates correct chat state."""
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        from ash.providers.telegram.handlers import TelegramMessageHandler

        handler = MagicMock(spec=TelegramMessageHandler)
        handler._provider = MagicMock()
        handler._provider.name = "telegram"

        TelegramMessageHandler._update_chat_state(
            handler, telegram_group_message, thread_id=None
        )

        # Verify state file
        state_path = (
            ash_home
            / "chats"
            / "telegram"
            / telegram_group_message.chat_id
            / "state.json"
        )
        assert state_path.exists()

        data = json.loads(state_path.read_text())
        assert data["chat"]["id"] == telegram_group_message.chat_id
        assert data["chat"]["type"] == "supergroup"
        assert data["chat"]["title"] == "My Test Group"
        assert len(data["participants"]) == 1
        assert data["participants"][0]["username"] == "bob"

    def test_thread_message_updates_state(
        self,
        monkeypatch,
        mock_get_chat_dir,
        ash_home: Path,
        telegram_thread_message: IncomingMessage,
    ):
        """Test that a thread message creates state in correct location."""
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        from ash.providers.telegram.handlers import TelegramMessageHandler

        handler = MagicMock(spec=TelegramMessageHandler)
        handler._provider = MagicMock()
        handler._provider.name = "telegram"

        thread_id = telegram_thread_message.metadata.get("thread_id")
        TelegramMessageHandler._update_chat_state(
            handler, telegram_thread_message, thread_id=thread_id
        )

        # Verify thread state file location
        state_path = (
            ash_home
            / "chats"
            / "telegram"
            / telegram_thread_message.chat_id
            / "threads"
            / "999"
            / "state.json"
        )
        assert state_path.exists()

        data = json.loads(state_path.read_text())
        assert data["participants"][0]["username"] == "charlie"

    def test_message_without_username(
        self,
        monkeypatch,
        mock_get_chat_dir,
        ash_home: Path,
        telegram_message_no_username: IncomingMessage,
    ):
        """Test handling messages from users without usernames."""
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        from ash.providers.telegram.handlers import TelegramMessageHandler

        handler = MagicMock(spec=TelegramMessageHandler)
        handler._provider = MagicMock()
        handler._provider.name = "telegram"

        TelegramMessageHandler._update_chat_state(
            handler, telegram_message_no_username, thread_id=None
        )

        state_path = (
            ash_home
            / "chats"
            / "telegram"
            / telegram_message_no_username.chat_id
            / "state.json"
        )
        data = json.loads(state_path.read_text())

        assert data["participants"][0]["id"] == "44444"
        assert data["participants"][0]["username"] is None
        assert data["participants"][0]["display_name"] == "No Username User"

    def test_multiple_users_in_group(
        self,
        monkeypatch,
        mock_get_chat_dir,
        ash_home: Path,
    ):
        """Test multiple users in a group are tracked correctly."""
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        from ash.providers.telegram.handlers import TelegramMessageHandler

        handler = MagicMock(spec=TelegramMessageHandler)
        handler._provider = MagicMock()
        handler._provider.name = "telegram"

        chat_id = "-100999888777"

        # Simulate messages from multiple users
        users = [
            ("11111", "alice", "Alice"),
            ("22222", "bob", "Bob"),
            ("33333", "charlie", "Charlie"),
            ("11111", "alice", "Alice"),  # Alice again
            ("22222", "bob", "Bob"),  # Bob again
        ]

        for user_id, username, display_name in users:
            message = IncomingMessage(
                id="msg-123",
                chat_id=chat_id,
                user_id=user_id,
                text="Hello",
                username=username,
                display_name=display_name,
                metadata={"chat_type": "supergroup", "chat_title": "Test Group"},
            )
            TelegramMessageHandler._update_chat_state(handler, message, thread_id=None)

        # Verify state
        state_path = ash_home / "chats" / "telegram" / chat_id / "state.json"
        data = json.loads(state_path.read_text())

        assert len(data["participants"]) == 3

        # Find participants by username
        by_username = {p["username"]: p for p in data["participants"]}
        assert by_username["alice"]["message_count"] == 2
        assert by_username["bob"]["message_count"] == 2
        assert by_username["charlie"]["message_count"] == 1


# =============================================================================
# State File Format Tests
# =============================================================================


class TestChatStateFileFormat:
    """Tests for the state file JSON format."""

    def test_state_json_serialization(
        self, monkeypatch, mock_get_chat_dir, ash_home: Path
    ):
        """Test that state is serialized correctly to JSON."""
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        manager = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
        )
        manager.update_chat_info(chat_type="supergroup", title="Test Group")
        manager.update_participant("user-1", "alice", "Alice Smith")

        state_path = ash_home / "chats" / "telegram" / "-123456" / "state.json"
        data = json.loads(state_path.read_text())

        # Verify required fields
        assert "chat" in data
        assert "participants" in data
        assert "updated_at" in data

        # Verify chat structure
        assert data["chat"]["id"] == "-123456"
        assert data["chat"]["type"] == "supergroup"
        assert data["chat"]["title"] == "Test Group"

        # Verify participant structure
        participant = data["participants"][0]
        assert participant["id"] == "user-1"
        assert participant["username"] == "alice"
        assert participant["display_name"] == "Alice Smith"
        assert participant["message_count"] == 1
        assert "first_seen" in participant
        assert "last_active" in participant

    def test_state_datetime_format(
        self, monkeypatch, mock_get_chat_dir, ash_home: Path
    ):
        """Test that datetimes are serialized in ISO format."""
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        manager = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
        )
        manager.update_participant("user-1", "alice", "Alice")

        state_path = ash_home / "chats" / "telegram" / "-123456" / "state.json"
        data = json.loads(state_path.read_text())

        # Verify datetime format is ISO
        updated_at = data["updated_at"]
        assert "T" in updated_at  # ISO format has T separator

        first_seen = data["participants"][0]["first_seen"]
        assert "T" in first_seen
