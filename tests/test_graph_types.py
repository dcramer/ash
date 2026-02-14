"""Tests for graph node and edge types."""

from datetime import UTC, datetime

from ash.graph.types import ChatEntry, EdgeType, UserEntry


class TestUserEntry:
    def test_to_dict_minimal(self):
        user = UserEntry(id="u1", provider="telegram", provider_id="123")
        d = user.to_dict()
        assert d["id"] == "u1"
        assert d["provider"] == "telegram"
        assert d["provider_id"] == "123"
        assert "username" not in d
        assert "display_name" not in d
        assert "person_id" not in d

    def test_to_dict_full(self):
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
        d = user.to_dict()
        assert d["username"] == "notzeeg"
        assert d["display_name"] == "David Cramer"
        assert d["person_id"] == "p1"
        assert d["metadata"] == {"source": "test"}

    def test_from_dict_minimal(self):
        d = {"id": "u1", "provider": "telegram", "provider_id": "123"}
        user = UserEntry.from_dict(d)
        assert user.id == "u1"
        assert user.provider == "telegram"
        assert user.provider_id == "123"
        assert user.username is None
        assert user.version == 1

    def test_from_dict_full(self):
        d = {
            "id": "u1",
            "version": 2,
            "provider": "telegram",
            "provider_id": "123",
            "username": "notzeeg",
            "display_name": "David Cramer",
            "person_id": "p1",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00Z",
            "metadata": {"source": "test"},
        }
        user = UserEntry.from_dict(d)
        assert user.version == 2
        assert user.username == "notzeeg"
        assert user.display_name == "David Cramer"
        assert user.person_id == "p1"
        assert user.created_at is not None
        assert user.updated_at is not None
        assert user.metadata == {"source": "test"}

    def test_roundtrip(self):
        now = datetime.now(UTC)
        user = UserEntry(
            id="u1",
            provider="telegram",
            provider_id="123",
            username="notzeeg",
            display_name="David",
            person_id="p1",
            created_at=now,
            updated_at=now,
        )
        d = user.to_dict()
        restored = UserEntry.from_dict(d)
        assert restored.id == user.id
        assert restored.provider == user.provider
        assert restored.provider_id == user.provider_id
        assert restored.username == user.username
        assert restored.display_name == user.display_name
        assert restored.person_id == user.person_id


class TestChatEntry:
    def test_to_dict_minimal(self):
        chat = ChatEntry(id="c1", provider="telegram", provider_id="456")
        d = chat.to_dict()
        assert d["id"] == "c1"
        assert d["provider"] == "telegram"
        assert d["provider_id"] == "456"
        assert "chat_type" not in d
        assert "title" not in d

    def test_to_dict_full(self):
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
        d = chat.to_dict()
        assert d["chat_type"] == "supergroup"
        assert d["title"] == "Test Group"

    def test_from_dict(self):
        d = {
            "id": "c1",
            "provider": "telegram",
            "provider_id": "456",
            "chat_type": "group",
            "title": "My Group",
        }
        chat = ChatEntry.from_dict(d)
        assert chat.id == "c1"
        assert chat.chat_type == "group"
        assert chat.title == "My Group"

    def test_roundtrip(self):
        now = datetime.now(UTC)
        chat = ChatEntry(
            id="c1",
            provider="telegram",
            provider_id="456",
            chat_type="supergroup",
            title="Test",
            created_at=now,
            updated_at=now,
        )
        d = chat.to_dict()
        restored = ChatEntry.from_dict(d)
        assert restored.id == chat.id
        assert restored.chat_type == chat.chat_type
        assert restored.title == chat.title


class TestEdgeType:
    def test_all_edge_types_exist(self):
        assert EdgeType.ABOUT.value == "about"
        assert EdgeType.OWNED_BY.value == "owned_by"
        assert EdgeType.IN_CHAT.value == "in_chat"
        assert EdgeType.STATED_BY.value == "stated_by"
        assert EdgeType.SUPERSEDES.value == "supersedes"
        assert EdgeType.KNOWS.value == "knows"
        assert EdgeType.IS_PERSON.value == "is_person"
        assert EdgeType.MERGED_INTO.value == "merged_into"
