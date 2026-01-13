"""Public types for the memory subsystem."""

from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    """Search result with similarity score."""

    id: str
    content: str
    similarity: float
    metadata: dict[str, Any] | None = None
    source_type: str = "memory"


@dataclass
class RetrievedContext:
    """Context retrieved from memory for LLM prompt augmentation."""

    memories: list[SearchResult]


@dataclass
class PersonResolutionResult:
    """Result of person resolution."""

    person_id: str
    created: bool
    person_name: str


@dataclass
class ExtractedFact:
    """A fact extracted from conversation."""

    content: str
    subjects: list[str]
    shared: bool
    confidence: float
