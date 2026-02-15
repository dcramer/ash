"""Deprecated: use ash.store.migration instead.

Backward-compatible re-exports.
"""

from ash.store.migration import migrate_filesystem

__all__ = [
    "migrate_filesystem",
]
