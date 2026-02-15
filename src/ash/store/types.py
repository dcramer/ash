"""Public types for the store subsystem.

Consolidates types from memory, people, and user/chat domains into a single module.
"""

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

    # Source Attribution (optional, for multi-user scenarios)
    source_username: str | None = None  # Who said/provided this fact (handle/username)
    source_display_name: str | None = None  # Display name for source user

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
    superseded_by_id: str | None = None

    # Archive fields (set when memory is archived)
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
        if self.source_username:
            d["source_username"] = self.source_username
        if self.source_display_name:
            d["source_display_name"] = self.source_display_name
        if self.source_session_id:
            d["source_session_id"] = self.source_session_id
        if self.source_message_id:
            d["source_message_id"] = self.source_message_id
        if self.extraction_confidence is not None:
            d["extraction_confidence"] = self.extraction_confidence
        if self.sensitivity is not None:
            d["sensitivity"] = self.sensitivity.value
        if not self.portable:
            d["portable"] = False
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
        """Deserialize from JSON dict.

        Supports both old field names (source_user_id, source_user_name)
        and new names (source_username, source_display_name) for backward compat.
        Also accepts embedding if present (for migration reading old format).
        """
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
            source_username=d.get("source_username") or d.get("source_user_id"),
            source_display_name=d.get("source_display_name")
            or d.get("source_user_name"),
            source_session_id=d.get("source_session_id"),
            source_message_id=d.get("source_message_id"),
            extraction_confidence=d.get("extraction_confidence"),
            sensitivity=Sensitivity(d["sensitivity"]) if d.get("sensitivity") else None,
            portable=d.get("portable", True),
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

    @staticmethod
    def serialize_embedding_bytes(floats: list[float]) -> bytes:
        """Serialize float list to bytes for sqlite-vec."""
        return struct.pack(f"{len(floats)}f", *floats)


@dataclass
class EmbeddingRecord:
    """Embedding storage record.

    Stores memory_id -> base64 embedding pairs separately from memories.
    """

    memory_id: str
    embedding: str  # Base64-encoded float32 array

    def to_dict(self) -> dict[str, Any]:
        return {"memory_id": self.memory_id, "embedding": self.embedding}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EmbeddingRecord":
        return cls(memory_id=d["memory_id"], embedding=d.get("embedding", ""))


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


@dataclass
class AliasEntry:
    """An alias for a person with provenance tracking."""

    value: str
    added_by: str | None = None
    created_at: datetime | None = None


@dataclass
class RelationshipClaim:
    """A relationship assertion with provenance tracking."""

    relationship: str
    stated_by: str | None = None
    created_at: datetime | None = None


@dataclass
class PersonEntry:
    """Person entity."""

    id: str
    version: int = 1
    created_by: str = ""
    name: str = ""
    relationships: list[RelationshipClaim] = field(default_factory=list)
    aliases: list[AliasEntry] = field(default_factory=list)
    merged_into: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d: dict[str, Any] = {
            "id": self.id,
            "version": self.version,
            "created_by": self.created_by,
            "name": self.name,
        }

        if self.relationships:
            d["relationships"] = [
                {
                    "relationship": rc.relationship,
                    **({"stated_by": rc.stated_by} if rc.stated_by else {}),
                    **(
                        {"created_at": rc.created_at.isoformat()}
                        if rc.created_at
                        else {}
                    ),
                }
                for rc in self.relationships
            ]
        if self.aliases:
            d["aliases"] = [
                {
                    "value": a.value,
                    **({"added_by": a.added_by} if a.added_by else {}),
                    **(
                        {"created_at": a.created_at.isoformat()} if a.created_at else {}
                    ),
                }
                for a in self.aliases
            ]
        if self.merged_into:
            d["merged_into"] = self.merged_into
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
        raw_aliases = d.get("aliases") or []
        aliases = [
            AliasEntry(
                value=a["value"] if isinstance(a, dict) else a,
                added_by=a.get("added_by") if isinstance(a, dict) else None,
                created_at=_parse_datetime(a.get("created_at"))
                if isinstance(a, dict)
                else None,
            )
            for a in raw_aliases
        ]

        raw_rels = d.get("relationships") or []
        relationships: list[RelationshipClaim]
        if isinstance(raw_rels, list):
            relationships = [
                RelationshipClaim(
                    relationship=r["relationship"] if isinstance(r, dict) else r,
                    stated_by=r.get("stated_by") if isinstance(r, dict) else None,
                    created_at=_parse_datetime(r.get("created_at"))
                    if isinstance(r, dict)
                    else None,
                )
                for r in raw_rels
            ]
        else:
            # Old format: single relationship string
            old_rel = d.get("relationship") or d.get("relation")
            relationships = [RelationshipClaim(relationship=old_rel)] if old_rel else []

        return cls(
            id=d["id"],
            version=d.get("version", 1),
            created_by=d.get("created_by") or d.get("owner_user_id", ""),
            name=d.get("name", ""),
            relationships=relationships,
            aliases=aliases,
            merged_into=d.get("merged_into"),
            created_at=_parse_datetime(d.get("created_at")),
            updated_at=_parse_datetime(d.get("updated_at")),
            metadata=d.get("metadata"),
        )

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


@dataclass
class UserEntry:
    """Provider user identity node.

    Bridges a provider-specific identity (Telegram user, etc.) to a Person record.
    The provider_id is the stable anchor; username and display_name can change.
    """

    id: str
    version: int = 1
    provider: str = ""  # "telegram", "cli", etc.
    provider_id: str = ""  # Stable provider user ID (e.g., "123456789")
    username: str | None = None  # Mostly-stable handle (e.g., "notzeeg")
    display_name: str | None = None  # Unstable display name
    person_id: str | None = None  # -> PersonEntry.id
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ChatEntry:
    """Chat/channel node.

    Represents a chat or channel from a provider.
    """

    id: str
    version: int = 1
    provider: str = ""  # "telegram", etc.
    provider_id: str = ""  # Provider chat ID
    chat_type: str | None = None  # "private", "group", "supergroup"
    title: str | None = None  # Group title (mutable)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] | None = None
