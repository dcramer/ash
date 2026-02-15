"""Deprecated: use ash.store instead.

Backward-compatible re-exports.
"""

from ash.store.people import RELATIONSHIP_TERMS as RELATIONSHIP_TERMS
from ash.store.store import Store as GraphStore
from ash.store.store import create_store as create_graph_store

__all__ = [
    "GraphStore",
    "RELATIONSHIP_TERMS",
    "create_graph_store",
]
