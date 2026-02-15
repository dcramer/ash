"""Deprecated: use ash.store.migration_sqlite instead.

Backward-compatible re-exports.
"""

from ash.store.migration_sqlite import migrate_jsonl_to_sqlite

__all__ = [
    "migrate_jsonl_to_sqlite",
]
