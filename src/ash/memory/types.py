"""Public types for the memory subsystem."""

import base64
import struct
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


def _parse_datetime(s: str | None) -> datetime | None:
    """Parse ISO datetime string, handling Z suffix and ensuring timezone awareness."""
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


class MemoryType(Enum):
    """Memory type classification for lifecycle management.

    Long-lived types (no automatic expiration):
    - preference: User likes, dislikes, preferences
    - identity: Facts about the user themselves
    - relationship: People in user's life and relationships
    - knowledge: Factual information about the world

    Ephemeral types (decay over time):
    - context: Current situation/state (7 days)
    - event: Past occurrences (30 days)
    - task: Things to do/remember (14 days)
    - observation: Fleeting observations (3 days)
    """

    PREFERENCE = "preference"
    IDENTITY = "identity"
    RELATIONSHIP = "relationship"
    KNOWLEDGE = "knowledge"
    CONTEXT = "context"
    EVENT = "event"
    TASK = "task"
    OBSERVATION = "observation"


# Types that decay over time without explicit expiration
EPHEMERAL_TYPES: set[MemoryType] = {
    MemoryType.CONTEXT,
    MemoryType.EVENT,
    MemoryType.TASK,
    MemoryType.OBSERVATION,
}

# Default TTL in days for ephemeral types
TYPE_TTL: dict[MemoryType, int] = {
    MemoryType.CONTEXT: 7,
    MemoryType.EVENT: 30,
    MemoryType.TASK: 14,
    MemoryType.OBSERVATION: 3,
}


@dataclass
class MemoryEntry:
    """Full memory entry schema for filesystem storage.

    Required fields:
    - id: UUID primary key
    - version: Schema version for migration
    - content: The memory text
    - memory_type: Type classification
    - embedding: Base64-encoded float32 array
    - created_at: When written to storage
    - source: Origin tracking (user, extraction, cli, rpc)
    - owner_user_id OR chat_id: At least one required for authorization

    Optional fields populated based on context.
    """

    # Identity (required)
    id: str
    version: int = 1

    # Content (required)
    content: str = ""
    memory_type: MemoryType = MemoryType.KNOWLEDGE
    embedding: str = ""  # Base64-encoded float32 array

    # Timestamps (required)
    created_at: datetime | None = None

    # Timestamps (optional)
    observed_at: datetime | None = (
        None  # When fact was observed (for delayed extraction)
    )

    # Scoping (at least one required for authorization)
    owner_user_id: str | None = None  # Personal scope
    chat_id: str | None = None  # Group scope

    # Subject (optional)
    subject_person_ids: list[str] = field(default_factory=list)

    # Source Attribution (required)
    source: str = "user"  # "user" | "extraction" | "cli" | "rpc"

    # Source Attribution (optional, for extraction tracing)
    source_session_id: str | None = None
    source_message_id: str | None = None
    extraction_confidence: float | None = None

    # Lifecycle (optional)
    expires_at: datetime | None = None
    superseded_at: datetime | None = None
    superseded_by_id: str | None = None

    # Archive fields (only present in archive.jsonl)
    archived_at: datetime | None = None
    archive_reason: str | None = None  # "superseded" | "expired" | "ephemeral_decay"

    # Extensibility
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d: dict[str, Any] = {
            "id": self.id,
            "version": self.version,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "embedding": self.embedding,
            "source": self.source,
        }

        # Add required timestamps
        if self.created_at:
            d["created_at"] = self.created_at.isoformat()

        # Add optional fields only if set
        if self.observed_at:
            d["observed_at"] = self.observed_at.isoformat()
        if self.owner_user_id:
            d["owner_user_id"] = self.owner_user_id
        if self.chat_id:
            d["chat_id"] = self.chat_id
        if self.subject_person_ids:
            d["subject_person_ids"] = self.subject_person_ids
        if self.source_session_id:
            d["source_session_id"] = self.source_session_id
        if self.source_message_id:
            d["source_message_id"] = self.source_message_id
        if self.extraction_confidence is not None:
            d["extraction_confidence"] = self.extraction_confidence
        if self.expires_at:
            d["expires_at"] = self.expires_at.isoformat()
        if self.superseded_at:
            d["superseded_at"] = self.superseded_at.isoformat()
        if self.superseded_by_id:
            d["superseded_by_id"] = self.superseded_by_id
        if self.archived_at:
            d["archived_at"] = self.archived_at.isoformat()
        if self.archive_reason:
            d["archive_reason"] = self.archive_reason
        if self.metadata:
            d["metadata"] = self.metadata

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryEntry":
        """Deserialize from JSON dict."""
        return cls(
            id=d["id"],
            version=d.get("version", 1),
            content=d.get("content", ""),
            memory_type=MemoryType(d.get("memory_type", "knowledge")),
            embedding=d.get("embedding", ""),
            created_at=_parse_datetime(d.get("created_at")),
            observed_at=_parse_datetime(d.get("observed_at")),
            owner_user_id=d.get("owner_user_id"),
            chat_id=d.get("chat_id"),
            subject_person_ids=d.get("subject_person_ids") or [],
            source=d.get("source", "user"),
            source_session_id=d.get("source_session_id"),
            source_message_id=d.get("source_message_id"),
            extraction_confidence=d.get("extraction_confidence"),
            expires_at=_parse_datetime(d.get("expires_at")),
            superseded_at=_parse_datetime(d.get("superseded_at")),
            superseded_by_id=d.get("superseded_by_id"),
            archived_at=_parse_datetime(d.get("archived_at")),
            archive_reason=d.get("archive_reason"),
            metadata=d.get("metadata"),
        )

    def get_embedding_bytes(self) -> bytes | None:
        """Decode base64 embedding to bytes for sqlite-vec."""
        if not self.embedding:
            return None
        return base64.b64decode(self.embedding)

    def get_embedding_floats(self) -> list[float] | None:
        """Decode base64 embedding to list of floats."""
        data = self.get_embedding_bytes()
        if not data:
            return None
        count = len(data) // 4
        return list(struct.unpack(f"{count}f", data))

    @staticmethod
    def encode_embedding(floats: list[float]) -> str:
        """Encode float list to base64 string."""
        data = struct.pack(f"{len(floats)}f", *floats)
        return base64.b64encode(data).decode("ascii")


@dataclass
class PersonEntry:
    """Person entity for filesystem storage."""

    id: str
    version: int = 1
    owner_user_id: str = ""
    name: str = ""
    relation: str | None = None
    aliases: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d: dict[str, Any] = {
            "id": self.id,
            "version": self.version,
            "owner_user_id": self.owner_user_id,
            "name": self.name,
        }

        if self.relation:
            d["relation"] = self.relation
        if self.aliases:
            d["aliases"] = self.aliases
        if self.created_at:
            d["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            d["updated_at"] = self.updated_at.isoformat()
        if self.metadata:
            d["metadata"] = self.metadata

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PersonEntry":
        """Deserialize from JSON dict."""
        return cls(
            id=d["id"],
            version=d.get("version", 1),
            owner_user_id=d.get("owner_user_id", ""),
            name=d.get("name", ""),
            relation=d.get("relation"),
            aliases=d.get("aliases") or [],
            created_at=_parse_datetime(d.get("created_at")),
            updated_at=_parse_datetime(d.get("updated_at")),
            metadata=d.get("metadata"),
        )


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
    memory_type: MemoryType = MemoryType.KNOWLEDGE


@dataclass
class GCResult:
    """Result of garbage collection."""

    removed_count: int
    archived_ids: list[str] = field(default_factory=list)
