"""Memory processing: extraction, embedding, indexing, secret detection.

Storage types are re-exported from ash.store for backward compatibility.

Public API:
- MemoryExtractor: Background extraction from conversations
- ExtractionPipeline: Full extraction workflow with person resolution

Types (from ash.store):
- MemoryEntry, MemoryType, GCResult, SearchResult, RetrievedContext, ExtractedFact
- matches_scope: Utility for scope filtering

Pipeline types:
- ResolvedFact: Fact with person IDs resolved
- ExtractionResult: Result of extraction pipeline

Internal:
- VectorIndex: SQLite-vec vector index
- EmbeddingGenerator: Embedding generation
"""

from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.extractor import MemoryExtractor
from ash.memory.index import VectorIndex
from ash.memory.pipeline import ExtractionPipeline, ExtractionResult, ResolvedFact
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
    "ExtractionPipeline",
    # Pipeline types
    "ResolvedFact",
    "ExtractionResult",
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
