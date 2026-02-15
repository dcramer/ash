"""Tests for store types (UserEntry, ChatEntry)."""

from datetime import UTC, datetime

from ash.store.types import ChatEntry, UserEntry


class TestUserEntry:
    def test_creation_minimal(self):
        user = UserEntry(id="u1", provider="telegram", provider_id="123")
        assert user.id == "u1"
        assert user.provider == "telegram"
        assert user.provider_id == "123"
        assert user.username is None
        assert user.display_name is None
        assert user.person_id is None
        assert user.version == 1

    def test_creation_full(self):
        now = datetime.now(UTC)
        user = UserEntry(
            id="u1",
            provider="telegram",
            provider_id="123",
            username="notzeeg",
            display_name="David Cramer",
            person_id="p1",
            created_at=now,
            updated_at=now,
            metadata={"source": "test"},
        )
        assert user.username == "notzeeg"
        assert user.display_name == "David Cramer"
        assert user.person_id == "p1"
        assert user.metadata == {"source": "test"}
        assert user.created_at == now
        assert user.updated_at == now


class TestChatEntry:
    def test_creation_minimal(self):
        chat = ChatEntry(id="c1", provider="telegram", provider_id="456")
        assert chat.id == "c1"
        assert chat.provider == "telegram"
        assert chat.provider_id == "456"
        assert chat.chat_type is None
        assert chat.title is None
        assert chat.version == 1

    def test_creation_full(self):
        now = datetime.now(UTC)
        chat = ChatEntry(
            id="c1",
            provider="telegram",
            provider_id="456",
            chat_type="supergroup",
            title="Test Group",
            created_at=now,
            updated_at=now,
        )
        assert chat.chat_type == "supergroup"
        assert chat.title == "Test Group"
        assert chat.created_at == now
        assert chat.updated_at == now
