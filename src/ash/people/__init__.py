"""Person/identity management.

Public API:
- GraphStore (from ash.graph): Primary facade for person operations

Types:
- PersonEntry: Person entity schema
- AliasEntry: Alias with provenance
- RelationshipClaim: Relationship with provenance
- PersonResolutionResult: Result of person lookup/creation
"""

from ash.people.types import (
    AliasEntry,
    PersonEntry,
    PersonResolutionResult,
    RelationshipClaim,
)

__all__ = [
    "AliasEntry",
    "PersonEntry",
    "PersonResolutionResult",
    "RelationshipClaim",
]
