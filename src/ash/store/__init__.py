"""Unified store for memory, people, users, and chats.

All data lives in one SQLite database. Vector search uses sqlite-vec.

Public API:
- Store: Unified facade for all operations
- create_store: Factory to create a wired Store

Mappers:
- row_to_memory, row_to_person, row_to_user, row_to_chat: Row conversion utilities

Types:
- MemoryEntry, MemoryType, Sensitivity, GCResult, SearchResult, RetrievedContext, ExtractedFact
- PersonEntry, AliasEntry, RelationshipClaim, PersonResolutionResult
- UserEntry, ChatEntry
- matches_scope: Utility for scope filtering
"""

from ash.store.mappers import (
    row_to_chat,
    row_to_memory,
    row_to_person,
    row_to_user,
)
from ash.store.store import Store, create_store
from ash.store.types import (
    AliasEntry,
    ChatEntry,
    EmbeddingRecord,
    ExtractedFact,
    GCResult,
    MemoryEntry,
    MemoryType,
    PersonEntry,
    PersonResolutionResult,
    RelationshipClaim,
    RetrievedContext,
    SearchResult,
    Sensitivity,
    UserEntry,
    matches_scope,
)

__all__ = [
    # Store
    "Store",
    "create_store",
    # Mappers
    "row_to_memory",
    "row_to_person",
    "row_to_user",
    "row_to_chat",
    # Memory types
    "MemoryEntry",
    "MemoryType",
    "Sensitivity",
    "GCResult",
    "ExtractedFact",
    "RetrievedContext",
    "SearchResult",
    "EmbeddingRecord",
    "matches_scope",
    # People types
    "PersonEntry",
    "AliasEntry",
    "RelationshipClaim",
    "PersonResolutionResult",
    # User/Chat types
    "UserEntry",
    "ChatEntry",
]
