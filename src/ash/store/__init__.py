"""Unified store for memory, people, users, and chats.

All data lives in an in-memory KnowledgeGraph backed by JSONL files.
Vector search uses numpy.

Public API:
- Store: Unified facade for all operations
- create_store: Factory to create a wired Store

Types:
- MemoryEntry, MemoryType, Sensitivity, GCResult, SearchResult, RetrievedContext, ExtractedFact
- PersonEntry, AliasEntry, RelationshipClaim, PersonResolutionResult
- UserEntry, ChatEntry
- matches_scope: Utility for scope filtering
"""

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
