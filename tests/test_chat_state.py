"""Tests for chat state management."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ash.chats import ChatInfo, ChatState, ChatStateManager, Participant
from ash.providers.base import IncomingMessage


@pytest.fixture
def telegram_private_message() -> IncomingMessage:
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
    ash_home = tmp_path / ".ash"
    ash_home.mkdir()
    return ash_home


@pytest.fixture
def mock_get_chat_dir(ash_home: Path):
    def _get_chat_dir(
        provider: str, chat_id: str, thread_id: str | None = None
    ) -> Path:
        base = ash_home / "chats" / provider / chat_id
        if thread_id:
            return base / "threads" / thread_id
        return base

    return _get_chat_dir


class TestChatStateModel:
    def test_create_default_state(self):
        state = ChatState(chat=ChatInfo(id="123"))

        assert state.chat.id == "123"
        assert state.participants == []
        assert state.updated_at is not None

    def test_create_state_with_chat_metadata(self):
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
        state = ChatState(chat=ChatInfo(id="123"))
        assert state.get_participant("user-1") is None

    def test_get_participant_found(self):
        state = ChatState(
            chat=ChatInfo(id="123"),
            participants=[
                Participant(
                    id="user-1",
                    username="alice",
                    display_name="Alice",
                )
            ],
        )

        participant = state.get_participant("user-1")
        assert participant is not None
        assert participant.username == "alice"

    def test_update_participant_creates_new(self):
        state = ChatState(chat=ChatInfo(id="123"))

        participant = state.update_participant(
            user_id="user-1",
            username="alice",
            display_name="Alice",
        )

        assert participant.id == "user-1"
        assert participant.username == "alice"
        assert participant.display_name == "Alice"
        assert participant.first_seen is not None
        assert participant.last_active is not None
        assert len(state.participants) == 1

    def test_update_participant_updates_existing(self):
        now = datetime.now(UTC)
        state = ChatState(
            chat=ChatInfo(id="123"),
            participants=[
                Participant(
                    id="user-1",
                    username="alice",
                    display_name="Alice",
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
        assert participant.first_seen == now  # Should not change
        assert participant.last_active >= now  # Should be updated
        assert len(state.participants) == 1

    def test_update_participant_preserves_none_fields(self):
        state = ChatState(
            chat=ChatInfo(id="123"),
            participants=[
                Participant(
                    id="user-1",
                    username="alice",
                    display_name="Alice",
                )
            ],
        )

        participant = state.update_participant(
            user_id="user-1",
            username=None,
            display_name="Alice Updated",
        )

        assert participant.username == "alice"
        assert participant.display_name == "Alice Updated"

    def test_multiple_participants(self):
        state = ChatState(chat=ChatInfo(id="group-123"))

        state.update_participant("user-1", "alice", "Alice")
        state.update_participant("user-2", "bob", "Bob")
        state.update_participant("user-3", None, "Charlie")

        assert len(state.participants) == 3
        assert state.get_participant("user-1").username == "alice"
        assert state.get_participant("user-2").username == "bob"
        assert state.get_participant("user-3").username is None


class TestChatStateManager:
    def test_creates_default_state(self, monkeypatch, mock_get_chat_dir):
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        manager = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
        )
        state = manager.load()

        assert state.chat.id == "-123456"
        assert state.participants == []

    def test_saves_and_loads_state(self, monkeypatch, mock_get_chat_dir):
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
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        manager = ChatStateManager(
            provider="telegram",
            chat_id="-100123456789",
        )
        manager.update_participant("user-1", "alice", "Alice")

        expected_path = ash_home / "chats" / "telegram" / "-100123456789" / "state.json"
        assert expected_path.exists()

        data = json.loads(expected_path.read_text())
        assert data["chat"]["id"] == "-100123456789"
        assert len(data["participants"]) == 1

    def test_thread_creates_separate_state(self, monkeypatch, mock_get_chat_dir):
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        chat_manager = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
        )
        chat_manager.update_participant("user-1", "alice", "Alice")

        thread_manager = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
            thread_id="789",
        )
        thread_manager.update_participant("user-2", "bob", "Bob")

        chat_state = chat_manager.load()
        thread_state = thread_manager.load()

        assert len(chat_state.participants) == 1
        assert chat_state.participants[0].id == "user-1"

        assert len(thread_state.participants) == 1
        assert thread_state.participants[0].id == "user-2"

    def test_thread_state_file_location(
        self, monkeypatch, mock_get_chat_dir, ash_home: Path
    ):
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
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        manager = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
        )

        manager.update_chat_info(chat_type="supergroup")
        state = manager.load()
        assert state.chat.type == "supergroup"
        assert state.chat.title is None

        manager.update_chat_info(title="My Group")
        state = manager.load()
        assert state.chat.type == "supergroup"
        assert state.chat.title == "My Group"

    def test_multiple_updates_same_participant(self, monkeypatch, mock_get_chat_dir):
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        manager = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
        )

        for _ in range(5):
            manager.update_participant("user-1", "alice", "Alice")

        state = manager.load()
        assert len(state.participants) == 1
        assert state.participants[0].username == "alice"


class TestTelegramChatStateIntegration:
    def test_private_message_updates_state(
        self,
        monkeypatch,
        mock_get_chat_dir,
        ash_home: Path,
        telegram_private_message: IncomingMessage,
    ):
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        from ash.providers.telegram.handlers import TelegramMessageHandler

        handler = MagicMock(spec=TelegramMessageHandler)
        handler._provider = MagicMock()
        handler._provider.name = "telegram"

        TelegramMessageHandler._update_chat_state(
            handler, telegram_private_message, thread_id=None
        )

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
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        from ash.providers.telegram.handlers import TelegramMessageHandler

        handler = MagicMock(spec=TelegramMessageHandler)
        handler._provider = MagicMock()
        handler._provider.name = "telegram"

        TelegramMessageHandler._update_chat_state(
            handler, telegram_group_message, thread_id=None
        )

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
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        from ash.providers.telegram.handlers import TelegramMessageHandler

        handler = MagicMock(spec=TelegramMessageHandler)
        handler._provider = MagicMock()
        handler._provider.name = "telegram"

        thread_id = telegram_thread_message.metadata.get("thread_id")
        TelegramMessageHandler._update_chat_state(
            handler, telegram_thread_message, thread_id=thread_id
        )

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
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        from ash.providers.telegram.handlers import TelegramMessageHandler

        handler = MagicMock(spec=TelegramMessageHandler)
        handler._provider = MagicMock()
        handler._provider.name = "telegram"

        chat_id = "-100999888777"
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

        state_path = ash_home / "chats" / "telegram" / chat_id / "state.json"
        data = json.loads(state_path.read_text())

        assert len(data["participants"]) == 3

        by_username = {p["username"]: p for p in data["participants"]}
        assert "alice" in by_username
        assert "bob" in by_username
        assert "charlie" in by_username


class TestChatStateFileFormat:
    def test_state_json_serialization(
        self, monkeypatch, mock_get_chat_dir, ash_home: Path
    ):
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        manager = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
        )
        manager.update_chat_info(chat_type="supergroup", title="Test Group")
        manager.update_participant("user-1", "alice", "Alice Smith")

        state_path = ash_home / "chats" / "telegram" / "-123456" / "state.json"
        data = json.loads(state_path.read_text())

        assert "chat" in data
        assert "participants" in data
        assert "updated_at" in data
        assert data["chat"]["id"] == "-123456"
        assert data["chat"]["type"] == "supergroup"
        assert data["chat"]["title"] == "Test Group"

        participant = data["participants"][0]
        assert participant["id"] == "user-1"
        assert participant["username"] == "alice"
        assert participant["display_name"] == "Alice Smith"
        assert "first_seen" in participant
        assert "last_active" in participant

    def test_state_datetime_format(
        self, monkeypatch, mock_get_chat_dir, ash_home: Path
    ):
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        manager = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
        )
        manager.update_participant("user-1", "alice", "Alice")

        state_path = ash_home / "chats" / "telegram" / "-123456" / "state.json"
        data = json.loads(state_path.read_text())

        assert "T" in data["updated_at"]
        assert "T" in data["participants"][0]["first_seen"]


class TestBidirectionalReferences:
    def test_participant_has_session_id(self, monkeypatch, mock_get_chat_dir):
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        manager = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
        )
        manager.update_participant(
            user_id="user-1",
            username="alice",
            display_name="Alice",
            session_id="telegram_-123456_user-1",
        )

        state = manager.load()
        participant = state.get_participant("user-1")
        assert participant.session_id == "telegram_-123456_user-1"

    def test_telegram_handler_sets_session_id(
        self,
        monkeypatch,
        mock_get_chat_dir,
        ash_home: Path,
        telegram_group_message: IncomingMessage,
    ):
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        from ash.providers.telegram.handlers import TelegramMessageHandler

        handler = MagicMock(spec=TelegramMessageHandler)
        handler._provider = MagicMock()
        handler._provider.name = "telegram"

        TelegramMessageHandler._update_chat_state(
            handler, telegram_group_message, thread_id=None
        )

        state_path = (
            ash_home
            / "chats"
            / "telegram"
            / telegram_group_message.chat_id
            / "state.json"
        )
        data = json.loads(state_path.read_text())

        participant = data["participants"][0]
        assert "session_id" in participant
        assert participant["session_id"].startswith("telegram_")
        assert telegram_group_message.chat_id.lstrip("-") in participant["session_id"]

    async def test_session_state_has_chat_reference(self, tmp_path: Path):
        from ash.sessions import SessionManager

        sessions_path = tmp_path / "sessions"

        manager = SessionManager(
            provider="telegram",
            chat_id="-100123456789",
            user_id="user-1",
            sessions_path=sessions_path,
        )

        await manager.ensure_session()

        state_path = manager.state_path
        assert state_path.exists()

        data = json.loads(state_path.read_text())
        assert data["provider"] == "telegram"
        assert data["chat_id"] == "-100123456789"
        assert data["user_id"] == "user-1"
        assert "created_at" in data

    async def test_session_state_with_thread_id(self, tmp_path: Path):
        from ash.sessions import SessionManager

        sessions_path = tmp_path / "sessions"

        manager = SessionManager(
            provider="telegram",
            chat_id="-100123456789",
            user_id="user-1",
            thread_id="999",
            sessions_path=sessions_path,
        )

        await manager.ensure_session()

        data = json.loads(manager.state_path.read_text())
        assert data["thread_id"] == "999"

    async def test_round_trip_references(
        self, monkeypatch, mock_get_chat_dir, ash_home: Path, tmp_path: Path
    ):
        monkeypatch.setattr("ash.chats.manager.get_chat_dir", mock_get_chat_dir)

        from ash.sessions import SessionManager, session_key

        sessions_path = tmp_path / "sessions"
        provider = "telegram"
        chat_id = "-100123456789"
        user_id = "user-1"

        session_manager = SessionManager(
            provider=provider,
            chat_id=chat_id,
            user_id=user_id,
            sessions_path=sessions_path,
        )
        await session_manager.ensure_session()

        sess_id = session_key(provider, chat_id, user_id)
        chat_state_manager = ChatStateManager(
            provider=provider,
            chat_id=chat_id,
        )
        chat_state_manager.update_participant(
            user_id=user_id,
            username="alice",
            session_id=sess_id,
        )

        chat_state = chat_state_manager.load()
        participant = chat_state.get_participant(user_id)
        assert participant.session_id == sess_id

        session_state_data = json.loads(session_manager.state_path.read_text())
        assert session_state_data["chat_id"] == chat_id
        assert session_state_data["provider"] == provider
