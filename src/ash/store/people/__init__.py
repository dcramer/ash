"""Person operations split into focused modules.

Re-exports PeopleOpsMixin for backward compatibility with store.py imports.
"""

from ash.store.people.crud import PeopleCrudMixin
from ash.store.people.dedup import PeopleDedupMixin
from ash.store.people.helpers import RELATIONSHIP_TERMS, normalize_reference
from ash.store.people.relationships import PeopleRelationshipsMixin
from ash.store.people.resolution import PeopleResolutionMixin


class PeopleOpsMixin(
    PeopleCrudMixin,
    PeopleRelationshipsMixin,
    PeopleResolutionMixin,
    PeopleDedupMixin,
):
    """Combined mixin for all people operations.

    Split implementation:
    - crud.py: create_person, get_person, update_person, delete_person, list_people
    - relationships.py: add_alias, add_relationship
    - resolution.py: find_person, find_person_for_speaker, resolve_or_create_person
    - dedup.py: merge_people, find_dedup_candidates, _follow_merge_chain
    """

    # Expose normalize_reference as static method for backward compatibility
    _normalize_reference = staticmethod(normalize_reference)


__all__ = [
    "PeopleOpsMixin",
    "RELATIONSHIP_TERMS",
    "normalize_reference",
]
