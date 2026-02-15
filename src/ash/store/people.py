"""Backward compatibility re-exports from store/people/ package.

All implementations have moved to store/people/*.py modules.
"""

from ash.store.people import (
    RELATIONSHIP_TERMS as RELATIONSHIP_TERMS,
)
from ash.store.people import (
    PeopleOpsMixin as PeopleOpsMixin,
)
from ash.store.people import (
    normalize_reference,
)

# Re-export helpers that were module-level in the old implementation
from ash.store.people.helpers import (
    FUZZY_MATCH_PROMPT,
)

__all__ = [
    "FUZZY_MATCH_PROMPT",
    "PeopleOpsMixin",
    "RELATIONSHIP_TERMS",
    "normalize_reference",
]
