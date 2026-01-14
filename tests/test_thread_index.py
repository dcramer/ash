"""Tests for ThreadIndex reply-chain tracking."""

import pytest

from ash.chats import ChatStateManager, ThreadIndex


@pytest.fixture
def thread_index(tmp_path, monkeypatch) -> ThreadIndex:
    """Create a ThreadIndex with isolated HOME directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    manager = ChatStateManager(provider="telegram", chat_id="-123456")
    return ThreadIndex(manager)


class TestThreadIndex:
    """Tests for ThreadIndex functionality."""

    def test_new_message_starts_thread(self, thread_index):
        """A new message without a reply starts its own thread."""
        thread_id = thread_index.resolve_thread_id("100", reply_to_external_id=None)
        assert thread_id == "100"

    def test_reply_joins_parent_thread(self, thread_index):
        """A reply to a known message joins that message's thread."""
        thread_id_100 = thread_index.resolve_thread_id("100", reply_to_external_id=None)
        thread_index.register_message("100", thread_id_100)

        thread_id_101 = thread_index.resolve_thread_id(
            "101", reply_to_external_id="100"
        )
        assert thread_id_101 == "100"

    def test_deep_reply_chain(self, thread_index):
        """A deep reply chain all shares the same thread."""
        # Build chain: 100 <- 101 <- 102 <- 103
        thread_id = thread_index.resolve_thread_id("100", reply_to_external_id=None)
        thread_index.register_message("100", thread_id)

        thread_id = thread_index.resolve_thread_id("101", reply_to_external_id="100")
        thread_index.register_message("101", thread_id)

        thread_id = thread_index.resolve_thread_id("102", reply_to_external_id="101")
        thread_index.register_message("102", thread_id)

        thread_id = thread_index.resolve_thread_id("103", reply_to_external_id="102")
        assert thread_id == "100"

    def test_reply_to_unknown_starts_new_thread(self, thread_index):
        """A reply to an unknown message starts a new thread."""
        thread_id = thread_index.resolve_thread_id("200", reply_to_external_id="999")
        assert thread_id == "200"

    def test_get_thread_id_registered(self, thread_index):
        """get_thread_id returns the thread for a registered message."""
        thread_index.register_message("100", "100")
        thread_index.register_message("101", "100")

        assert thread_index.get_thread_id("100") == "100"
        assert thread_index.get_thread_id("101") == "100"

    def test_get_thread_id_unregistered(self, thread_index):
        """get_thread_id returns None for unregistered message."""
        assert thread_index.get_thread_id("999") is None

    def test_persistence(self, tmp_path, monkeypatch):
        """Thread index persists across instances."""
        monkeypatch.setenv("HOME", str(tmp_path))

        # First instance
        manager1 = ChatStateManager(provider="telegram", chat_id="-123456")
        index1 = ThreadIndex(manager1)
        index1.register_message("100", "100")
        index1.register_message("101", "100")

        # Second instance (new manager, should load from disk)
        manager2 = ChatStateManager(provider="telegram", chat_id="-123456")
        index2 = ThreadIndex(manager2)

        assert index2.get_thread_id("100") == "100"
        assert index2.get_thread_id("101") == "100"

    def test_multiple_independent_threads(self, thread_index):
        """Multiple independent conversations have separate threads."""
        # Thread 1: messages 100, 101, 102
        thread_1 = thread_index.resolve_thread_id("100", reply_to_external_id=None)
        thread_index.register_message("100", thread_1)
        thread_id = thread_index.resolve_thread_id("101", reply_to_external_id="100")
        thread_index.register_message("101", thread_id)
        thread_id = thread_index.resolve_thread_id("102", reply_to_external_id="101")
        thread_index.register_message("102", thread_id)

        # Thread 2: messages 200, 201 (unrelated to thread 1)
        thread_2 = thread_index.resolve_thread_id("200", reply_to_external_id=None)
        thread_index.register_message("200", thread_2)
        thread_id = thread_index.resolve_thread_id("201", reply_to_external_id="200")
        thread_index.register_message("201", thread_id)

        # Verify threads are separate
        assert thread_index.get_thread_id("100") == "100"
        assert thread_index.get_thread_id("101") == "100"
        assert thread_index.get_thread_id("102") == "100"
        assert thread_index.get_thread_id("200") == "200"
        assert thread_index.get_thread_id("201") == "200"
