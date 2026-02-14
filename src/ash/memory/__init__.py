"""Memory and retrieval system.

Public API:
- GraphStore (from ash.graph): Primary facade for all memory operations
- MemoryExtractor: Background extraction from conversations

Types:
- MemoryEntry: Full memory entry schema
- MemoryType: Memory type classification
- GCResult: Result of garbage collection
- RetrievedContext: Context for LLM augmentation
- SearchResult: Individual search result with similarity
- ExtractedFact: Fact extracted from conversation
- matches_scope: Utility for scope filtering

Internal (not part of public API, may change):
- FileMemoryStore: JSONL-based memory store (use GraphStore instead)
- VectorIndex: SQLite-vec vector index (use GraphStore instead)
- EmbeddingGenerator: Embedding generation (internal)
"""

from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.extractor import MemoryExtractor
from ash.memory.file_store import FileMemoryStore
from ash.memory.index import VectorIndex
from ash.memory.types import (
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
    # Types
    "MemoryEntry",
    "MemoryType",
    "GCResult",
    "ExtractedFact",
    "RetrievedContext",
    "SearchResult",
    "matches_scope",
    # Internal (exposed for advanced use, prefer GraphStore)
    "FileMemoryStore",
    "VectorIndex",
    "EmbeddingGenerator",
]
