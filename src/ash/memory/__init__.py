"""Memory processing: extraction, embedding, indexing, secret detection.

Storage types are re-exported from ash.store for backward compatibility.

Public API:
- MemoryExtractor: Background extraction from conversations

Types (from ash.store):
- MemoryEntry, MemoryType, GCResult, SearchResult, RetrievedContext, ExtractedFact
- matches_scope: Utility for scope filtering

Internal:
- VectorIndex: SQLite-vec vector index
- EmbeddingGenerator: Embedding generation
"""

from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.extractor import MemoryExtractor
from ash.memory.index import VectorIndex
from ash.store.types import (
    ExtractedFact,
    GCResult,
    MemoryEntry,
    MemoryType,
    RetrievedContext,
    SearchResult,
    matches_scope,
)

__all__ = [
    # Extraction
    "MemoryExtractor",
    # Types (re-exported from ash.store)
    "MemoryEntry",
    "MemoryType",
    "GCResult",
    "ExtractedFact",
    "RetrievedContext",
    "SearchResult",
    "matches_scope",
    # Internal
    "VectorIndex",
    "EmbeddingGenerator",
]
