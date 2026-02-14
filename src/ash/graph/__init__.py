"""Unified graph architecture for memory, people, users, and chats.

Public API:
- GraphStore: Unified facade replacing MemoryManager + PersonManager
- create_graph_store: Factory to create wired GraphStore

Types:
- UserEntry: Provider user identity node
- ChatEntry: Chat/channel node
- EdgeType: Edge type enumeration

Internal:
- GraphIndex: In-memory adjacency list index (use GraphStore instead)
"""

from ash.graph.store import GraphStore, create_graph_store
from ash.graph.types import ChatEntry, EdgeType, UserEntry

__all__ = [
    "ChatEntry",
    "EdgeType",
    "GraphStore",
    "UserEntry",
    "create_graph_store",
]
