"""Tests for chat state management."""

from pathlib import Path

from ash.chats import ChatInfo, ChatState, ChatStateManager, Participant


class TestChatStateModel:
    """Tests for ChatState model."""

    def test_create_default_state(self):
        """Test creating a default chat state."""
        state = ChatState(chat=ChatInfo(id="123"))

        assert state.chat.id == "123"
        assert state.participants == []
        assert state.updated_at is not None

    def test_get_participant_not_found(self):
        """Test getting non-existent participant returns None."""
        state = ChatState(chat=ChatInfo(id="123"))

        assert state.get_participant("user-1") is None

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
        assert len(state.participants) == 1

    def test_update_participant_updates_existing(self):
        """Test update_participant updates existing participant."""
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

        participant = state.update_participant(
            user_id="user-1",
            username="alice_new",
            display_name="Alice Smith",
        )

        assert participant.username == "alice_new"
        assert participant.display_name == "Alice Smith"
        assert participant.message_count == 6
        assert len(state.participants) == 1


class TestChatStateManager:
    """Tests for ChatStateManager."""

    def test_creates_default_state(self, tmp_path: Path, monkeypatch):
        """Test manager creates default state for new chat."""
        monkeypatch.setattr(
            "ash.chats.manager.get_chat_dir",
            lambda p, c, t=None: tmp_path / p / c,
        )

        manager = ChatStateManager(
            provider="telegram",
            chat_id="-123456",
        )
        state = manager.load()

        assert state.chat.id == "-123456"
        assert state.participants == []

    def test_saves_and_loads_state(self, tmp_path: Path, monkeypatch):
        """Test state persistence."""
        monkeypatch.setattr(
            "ash.chats.manager.get_chat_dir",
            lambda p, c, t=None: tmp_path / p / c,
        )

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

    def test_thread_creates_separate_state(self, tmp_path: Path, monkeypatch):
        """Test thread has separate state from parent chat."""

        def mock_get_chat_dir(p, c, t=None):
            if t:
                return tmp_path / p / c / "threads" / t
            return tmp_path / p / c

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
