"""Deprecated: use ash.store.people instead.

Backward-compatible re-exports.
"""

from ash.store.people import RELATIONSHIP_TERMS, PeopleOpsMixin

__all__ = [
    "PeopleOpsMixin",
    "RELATIONSHIP_TERMS",
]
