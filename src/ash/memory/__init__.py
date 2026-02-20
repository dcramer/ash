"""Memory processing: extraction, embedding, indexing, secret detection.

Storage types are re-exported from ash.store for backward compatibility.

Public API:
- MemoryExtractor: Background extraction from conversations

Types (from ash.store):
- MemoryEntry, MemoryType, GCResult, SearchResult, RetrievedContext, ExtractedFact
- matches_scope: Utility for scope filtering

Internal:
- EmbeddingGenerator: Embedding generation
"""

from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.extractor import MemoryExtractor
from ash.store.types import (
    AssertionEnvelope,
    AssertionKind,
    AssertionPredicate,
    ExtractedFact,
    GCResult,
    MemoryEntry,
    MemoryType,
    PredicateObjectType,
    RetrievedContext,
    SearchResult,
    assertion_metadata_summary,
    get_assertion,
    matches_scope,
    upsert_assertion_metadata,
)

__all__ = [
    # Extraction
    "MemoryExtractor",
    # Types (re-exported from ash.store)
    "MemoryEntry",
    "MemoryType",
    "AssertionEnvelope",
    "AssertionKind",
    "AssertionPredicate",
    "PredicateObjectType",
    "GCResult",
    "ExtractedFact",
    "RetrievedContext",
    "SearchResult",
    "get_assertion",
    "upsert_assertion_metadata",
    "assertion_metadata_summary",
    "matches_scope",
    # Internal
    "EmbeddingGenerator",
]
