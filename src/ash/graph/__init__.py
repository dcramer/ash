"""Deprecated: use ash.store instead.

Backward-compatible re-exports.
"""

from ash.store import ChatEntry, Store, UserEntry, create_store

# Backward compat aliases
GraphStore = Store
create_graph_store = create_store

__all__ = [
    "ChatEntry",
    "GraphStore",
    "UserEntry",
    "create_graph_store",
]
