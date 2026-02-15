"""Chat state management.

Tracks chat metadata and participants separately from conversation history.
"""

from ash.chats.history import ChatHistoryWriter, HistoryEntry, read_recent_chat_history
from ash.chats.manager import ChatStateManager
from ash.chats.models import ChatInfo, ChatState, Participant
from ash.chats.thread_index import ThreadIndex

__all__ = [
    "ChatHistoryWriter",
    "ChatInfo",
    "ChatState",
    "ChatStateManager",
    "HistoryEntry",
    "Participant",
    "ThreadIndex",
    "read_recent_chat_history",
]
