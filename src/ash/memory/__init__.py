"""Memory and retrieval system.

Public API:
- MemoryManager: Primary facade for all memory operations
- create_memory_manager: Factory to create wired MemoryManager
- MemoryExtractor: Background extraction from conversations

Types:
- RetrievedContext: Context for LLM augmentation
- SearchResult: Individual search result with similarity
- PersonResolutionResult: Result of person lookup/creation
- ExtractedFact: Fact extracted from conversation

Internal (exposed for composition):
- MemoryStore, SemanticRetriever, EmbeddingGenerator
"""

from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.extractor import MemoryExtractor
from ash.memory.manager import MemoryManager, create_memory_manager
from ash.memory.retrieval import SemanticRetriever
from ash.memory.store import MemoryStore
from ash.memory.types import (
    ExtractedFact,
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
    "ExtractedFact",
    "PersonResolutionResult",
    "RetrievedContext",
    "SearchResult",
    # Internal components
    "MemoryStore",
    "SemanticRetriever",
    "EmbeddingGenerator",
]
