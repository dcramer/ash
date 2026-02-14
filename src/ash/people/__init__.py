"""Person/identity management.

Public API:
- PersonManager: Primary facade for person operations
- create_person_manager: Factory function

Types:
- PersonEntry: Person entity schema
- AliasEntry: Alias with provenance
- RelationshipClaim: Relationship with provenance
- PersonResolutionResult: Result of person lookup/creation
"""

from ash.people.manager import PersonManager, create_person_manager
from ash.people.types import (
    AliasEntry,
    PersonEntry,
    PersonResolutionResult,
    RelationshipClaim,
)

__all__ = [
    "AliasEntry",
    "PersonManager",
    "PersonEntry",
    "PersonResolutionResult",
    "RelationshipClaim",
    "create_person_manager",
]
