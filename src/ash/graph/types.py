"""Graph node and edge types for the unified graph architecture."""

from dataclasses import dataclass
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


class EdgeType(Enum):
    """Edge types in the graph.

    Each edge connects two node IDs with a semantic relationship.
    Edges are stored as fields on nodes and extracted by GraphIndex at build time.
    """

    # Memory edges
    ABOUT = "about"  # Memory -> Person (subject_person_ids)
    OWNED_BY = "owned_by"  # Memory -> User (owner_user_id via provider_id)
    IN_CHAT = "in_chat"  # Memory -> Chat (chat_id via provider_id)
    STATED_BY = "stated_by"  # Memory -> User (source_username via username)
    SUPERSEDES = "supersedes"  # Memory -> Memory (superseded_by_id)

    # Person edges
    KNOWS = "knows"  # User -> Person (relationships[].stated_by)
    IS_PERSON = "is_person"  # User -> Person (user.person_id)
    MERGED_INTO = "merged_into"  # Person -> Person (person.merged_into)


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
    person_id: str | None = None  # IS_PERSON edge -> PersonEntry.id
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "version": self.version,
            "provider": self.provider,
            "provider_id": self.provider_id,
        }
        if self.username:
            d["username"] = self.username
        if self.display_name:
            d["display_name"] = self.display_name
        if self.person_id:
            d["person_id"] = self.person_id
        if self.created_at:
            d["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            d["updated_at"] = self.updated_at.isoformat()
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UserEntry":
        return cls(
            id=d["id"],
            version=d.get("version", 1),
            provider=d.get("provider", ""),
            provider_id=d.get("provider_id", ""),
            username=d.get("username"),
            display_name=d.get("display_name"),
            person_id=d.get("person_id"),
            created_at=_parse_datetime(d.get("created_at")),
            updated_at=_parse_datetime(d.get("updated_at")),
            metadata=d.get("metadata"),
        )


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

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "version": self.version,
            "provider": self.provider,
            "provider_id": self.provider_id,
        }
        if self.chat_type:
            d["chat_type"] = self.chat_type
        if self.title:
            d["title"] = self.title
        if self.created_at:
            d["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            d["updated_at"] = self.updated_at.isoformat()
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChatEntry":
        return cls(
            id=d["id"],
            version=d.get("version", 1),
            provider=d.get("provider", ""),
            provider_id=d.get("provider_id", ""),
            chat_type=d.get("chat_type"),
            title=d.get("title"),
            created_at=_parse_datetime(d.get("created_at")),
            updated_at=_parse_datetime(d.get("updated_at")),
            metadata=d.get("metadata"),
        )
