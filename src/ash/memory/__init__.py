"""Memory and retrieval system.

Public API:
- MemoryManager: Primary facade for all memory operations
- create_memory_manager: Factory to create wired MemoryManager
- MemoryExtractor: Background extraction from conversations

Types:
- MemoryEntry: Full memory entry schema
- MemoryType: Memory type classification
- PersonEntry: Person entity schema
- GCResult: Result of garbage collection
- RetrievedContext: Context for LLM augmentation
- SearchResult: Individual search result with similarity
- PersonResolutionResult: Result of person lookup/creation
- ExtractedFact: Fact extracted from conversation

Storage (filesystem-primary):
- FileMemoryStore: JSONL-based memory store
- VectorIndex: SQLite-vec vector index

Internal (exposed for composition):
- EmbeddingGenerator
"""

from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.extractor import MemoryExtractor
from ash.memory.file_store import FileMemoryStore
from ash.memory.index import VectorIndex
from ash.memory.manager import MemoryManager, create_memory_manager
from ash.memory.types import (
    ExtractedFact,
    GCResult,
    MemoryEntry,
    MemoryType,
    PersonEntry,
    PersonResolutionResult,
    RetrievedContext,
    SearchResult,
)

__all__ = [
    # Primary facade
    "MemoryManager",
    "create_memory_manager",
    # Extraction
    "MemoryExtractor",
    # Types
    "MemoryEntry",
    "MemoryType",
    "PersonEntry",
    "GCResult",
    "ExtractedFact",
    "PersonResolutionResult",
    "RetrievedContext",
    "SearchResult",
    # Storage components
    "FileMemoryStore",
    "VectorIndex",
    # Internal components
    "EmbeddingGenerator",
]
