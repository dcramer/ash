"""Public types for the store subsystem.

Consolidates types from memory, people, and user/chat domains into a single module.
"""

import base64
import struct
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# =============================================================================
# Memory Types
# =============================================================================


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


class Sensitivity(Enum):
    """Sensitivity classification for privacy-aware retrieval.

    Controls when memories can be shared based on context:
    - PUBLIC: Can be shared anywhere (default)
    - PERSONAL: Only shown to the subject person or memory owner
    - SENSITIVE: High privacy (medical, financial) - only in private chat with subject
    """

    PUBLIC = "public"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"


class AssertionKind(Enum):
    """Semantic assertion class for a memory."""

    SELF_FACT = "self_fact"
    PERSON_FACT = "person_fact"
    RELATIONSHIP_FACT = "relationship_fact"
    GROUP_FACT = "group_fact"
    CONTEXT_FACT = "context_fact"


class PredicateObjectType(Enum):
    """Object type for structured assertion predicates."""

    TEXT = "text"
    PERSON = "person"
    TIME = "time"
    ENUM = "enum"


class AssertionPredicate(BaseModel):
    """Structured predicate for assertion metadata."""

    model_config = ConfigDict(frozen=False)

    name: str
    object_type: PredicateObjectType
    value: str


class AssertionEnvelope(BaseModel):
    """Canonical semantic envelope for memory assertions."""

    model_config = ConfigDict(frozen=False)

    semantic_version: int = 1
    assertion_kind: AssertionKind
    subjects: list[str] = Field(default_factory=list)
    speaker_person_id: str | None = None
    predicates: list[AssertionPredicate] = Field(default_factory=list)
    confidence: float = 1.0

    @field_validator("subjects", mode="before")
    @classmethod
    def _coerce_subjects(cls, v: list[str] | None) -> list[str]:
        return [s for s in (v or []) if s]

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v


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


class MemoryEntry(BaseModel):
    """Full memory entry schema.

    Required fields:
    - id: UUID primary key
    - version: Schema version for migration
    - content: The memory text
    - memory_type: Type classification
    - created_at: When written to storage
    - source: Origin tracking (user, extraction, cli, rpc)
    - owner_user_id OR chat_id: At least one required for authorization

    Optional fields populated based on context.
    """

    model_config = ConfigDict(frozen=False)

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
    observed_at: datetime | None = None

    # Scoping (at least one required for authorization)
    owner_user_id: str | None = None  # Personal scope
    chat_id: str | None = None  # Group scope

    # Source Attribution (required)
    source: str = "user"  # "user" | "extraction" | "cli" | "rpc"

    # Source Attribution (optional, for multi-user scenarios)
    source_username: str | None = None
    source_display_name: str | None = None

    # Source Attribution (optional, for extraction tracing)
    source_session_id: str | None = None
    source_message_id: str | None = None
    extraction_confidence: float | None = None

    # Privacy (optional)
    sensitivity: Sensitivity | None = None  # None = PUBLIC for backward compat

    # Cross-context portability (optional)
    portable: bool = True  # Whether this memory crosses chat boundaries via ABOUT edges

    # Lifecycle (optional)
    expires_at: datetime | None = None
    superseded_at: datetime | None = None

    # Archive fields (set when memory is archived)
    archived_at: datetime | None = None
    archive_reason: str | None = None  # "superseded" | "expired" | "ephemeral_decay"

    # Extensibility
    metadata: dict[str, Any] | None = None

    def is_active(self, now: datetime | None = None) -> bool:
        """Return True if memory is not archived, not superseded, and not expired.

        If ``now`` is omitted the expiry check is skipped.
        """
        if self.archived_at is not None:
            return False
        if self.superseded_at is not None:
            return False
        if now and self.expires_at and self.expires_at <= now:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d = self.model_dump(mode="json", exclude_none=True, exclude={"embedding"})
        # portable=True is the default; only serialize when False
        if d.get("portable") is True:
            d.pop("portable")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryEntry":
        """Deserialize from JSON dict."""
        return cls.model_validate(d)

    @staticmethod
    def encode_embedding(floats: list[float]) -> str:
        """Encode float list to base64 string."""
        data = struct.pack(f"{len(floats)}f", *floats)
        return base64.b64encode(data).decode("ascii")


def get_assertion(memory: MemoryEntry) -> AssertionEnvelope | None:
    """Read and parse assertion envelope from memory metadata."""
    if not memory.metadata:
        return None

    raw = memory.metadata.get("assertion")
    if not isinstance(raw, dict):
        return None

    try:
        return AssertionEnvelope.model_validate(raw)
    except Exception:
        return None


def upsert_assertion_metadata(
    metadata: dict[str, Any] | None,
    assertion: AssertionEnvelope,
) -> dict[str, Any]:
    """Write assertion envelope into metadata with semantic version."""
    result = dict(metadata or {})
    result["semantic_version"] = assertion.semantic_version
    result["assertion"] = assertion.model_dump(mode="json", exclude_none=True)
    return result


def assertion_metadata_summary(memory: MemoryEntry) -> dict[str, Any]:
    """Return compact assertion summary fields for retrieval metadata."""
    assertion = get_assertion(memory)
    if not assertion:
        return {}

    summary: dict[str, Any] = {
        "semantic_version": assertion.semantic_version,
        "assertion_kind": assertion.assertion_kind.value,
    }
    if assertion.subjects:
        summary["assertion_subject_ids"] = assertion.subjects
    if assertion.speaker_person_id:
        summary["speaker_person_id"] = assertion.speaker_person_id
    return summary


class EmbeddingRecord(BaseModel):
    """Embedding storage record.

    Stores memory_id -> base64 embedding pairs separately from memories.
    """

    model_config = ConfigDict(frozen=False)

    memory_id: str
    embedding: str = ""  # Base64-encoded float32 array

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EmbeddingRecord":
        return cls.model_validate(d)


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
class ExtractedFact:
    """A fact extracted from conversation."""

    content: str
    subjects: list[str]
    shared: bool
    confidence: float
    memory_type: MemoryType = MemoryType.KNOWLEDGE
    speaker: str | None = None  # Who said this (username or identifier)
    sensitivity: Sensitivity | None = None  # Privacy classification
    portable: bool = True  # Whether this fact crosses chat boundaries
    assertion: AssertionEnvelope | None = None  # Optional structured semantics override
    aliases: dict[str, list[str]] = field(
        default_factory=dict
    )  # subject name -> alias strings


@dataclass
class GCResult:
    """Result of garbage collection."""

    removed_count: int
    archived_ids: list[str] = field(default_factory=list)


def matches_scope(
    memory: MemoryEntry,
    owner_user_id: str | None = None,
    chat_id: str | None = None,
) -> bool:
    """Check if a memory matches the given scope filters.

    Memory scoping rules:
    - Personal: owner_user_id set - only visible to that user
    - Group: owner_user_id NULL, chat_id set - visible to everyone in that chat
    """
    if not owner_user_id and not chat_id:
        return True
    if owner_user_id and memory.owner_user_id == owner_user_id:
        return True
    if chat_id and memory.owner_user_id is None and memory.chat_id == chat_id:
        return True
    return False


# =============================================================================
# People Types
# =============================================================================


class AliasEntry(BaseModel):
    """An alias for a person with provenance tracking."""

    model_config = ConfigDict(frozen=False)

    value: str
    added_by: str | None = None
    created_at: datetime | None = None


class RelationshipClaim(BaseModel):
    """A relationship assertion with provenance tracking."""

    model_config = ConfigDict(frozen=False)

    relationship: str
    stated_by: str | None = None
    created_at: datetime | None = None


class PersonEntry(BaseModel):
    """Person entity."""

    model_config = ConfigDict(frozen=False)

    id: str
    version: int = 1
    created_by: str = ""
    name: str = ""
    relationships: list[RelationshipClaim] = []
    aliases: list[AliasEntry] = []
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("aliases", mode="before")
    @classmethod
    def _coerce_aliases(cls, v: list | None) -> list:
        """Handle legacy format where aliases were plain strings."""
        return [{"value": a} if isinstance(a, str) else a for a in (v or [])]

    @field_validator("relationships", mode="before")
    @classmethod
    def _coerce_relationships(cls, v: list | None) -> list:
        """Handle legacy format where relationships were plain strings."""
        return [{"relationship": r} if isinstance(r, str) else r for r in (v or [])]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d = self.model_dump(mode="json", exclude_none=True)
        # Don't serialize empty lists
        if not d.get("relationships"):
            d.pop("relationships", None)
        if not d.get("aliases"):
            d.pop("aliases", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PersonEntry":
        """Deserialize from JSON dict."""
        return cls.model_validate(d)

    def matches_username(self, username: str) -> bool:
        """Check if this person matches a username (case-insensitive)."""
        username_lower = username.lower()
        if self.name and self.name.lower() == username_lower:
            return True
        return any(alias.value.lower() == username_lower for alias in self.aliases)


@dataclass
class PersonResolutionResult:
    """Result of person resolution."""

    person_id: str
    created: bool
    person_name: str


# =============================================================================
# User/Chat Types
# =============================================================================


class UserEntry(BaseModel):
    """Provider user identity node.

    Bridges a provider-specific identity (Telegram user, etc.) to a Person record.
    The provider_id is the stable anchor; username and display_name can change.
    """

    model_config = ConfigDict(frozen=False)

    id: str
    version: int = 1
    provider: str = ""  # "telegram", "cli", etc.
    provider_id: str = ""  # Stable provider user ID (e.g., "123456789")
    username: str | None = None  # Mostly-stable handle (e.g., "notzeeg")
    display_name: str | None = None  # Unstable display name
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UserEntry":
        return cls.model_validate(d)


class ChatEntry(BaseModel):
    """Chat/channel node.

    Represents a chat or channel from a provider.
    """

    model_config = ConfigDict(frozen=False)

    id: str
    version: int = 1
    provider: str = ""  # "telegram", etc.
    provider_id: str = ""  # Provider chat ID
    chat_type: str | None = None  # "private", "group", "supergroup"
    title: str | None = None  # Group title (mutable)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChatEntry":
        return cls.model_validate(d)
