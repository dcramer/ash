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
    AssertionEnvelope,
    AssertionKind,
    AssertionPredicate,
    ChatEntry,
    EmbeddingRecord,
    ExtractedFact,
    GCResult,
    MemoryEntry,
    MemoryType,
    PersonEntry,
    PersonResolutionResult,
    PredicateObjectType,
    RelationshipClaim,
    RetrievedContext,
    SearchResult,
    Sensitivity,
    UserEntry,
    assertion_metadata_summary,
    get_assertion,
    matches_scope,
    upsert_assertion_metadata,
)

__all__ = [
    # Store
    "Store",
    "create_store",
    # Memory types
    "MemoryEntry",
    "MemoryType",
    "Sensitivity",
    "AssertionEnvelope",
    "AssertionKind",
    "AssertionPredicate",
    "PredicateObjectType",
    "GCResult",
    "ExtractedFact",
    "RetrievedContext",
    "SearchResult",
    "EmbeddingRecord",
    "get_assertion",
    "upsert_assertion_metadata",
    "assertion_metadata_summary",
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
