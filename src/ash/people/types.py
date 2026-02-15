"""Public types for the people subsystem."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
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


@dataclass
class AliasEntry:
    """An alias for a person with provenance tracking."""

    value: str
    added_by: str | None = None
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"value": self.value}
        if self.added_by:
            d["added_by"] = self.added_by
        if self.created_at:
            d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AliasEntry":
        return cls(
            value=d["value"],
            added_by=d.get("added_by"),
            created_at=_parse_datetime(d.get("created_at")),
        )


@dataclass
class RelationshipClaim:
    """A relationship assertion with provenance tracking."""

    relationship: str
    stated_by: str | None = None
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"relationship": self.relationship}
        if self.stated_by:
            d["stated_by"] = self.stated_by
        if self.created_at:
            d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RelationshipClaim":
        return cls(
            relationship=d["relationship"],
            stated_by=d.get("stated_by"),
            created_at=_parse_datetime(d.get("created_at")),
        )


@dataclass
class PersonEntry:
    """Person entity for filesystem storage."""

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
            d["relationships"] = [r.to_dict() for r in self.relationships]
        if self.aliases:
            d["aliases"] = [a.to_dict() for a in self.aliases]
        if self.merged_into:
            d["merged_into"] = self.merged_into
        if self.created_at:
            d["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            d["updated_at"] = self.updated_at.isoformat()
        if self.metadata:
            d["metadata"] = self.metadata

        return d

    def matches_username(self, username: str) -> bool:
        """Check if this person matches a username (case-insensitive)."""
        username_lower = username.lower()
        if self.name and self.name.lower() == username_lower:
            return True
        return any(alias.value.lower() == username_lower for alias in self.aliases)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PersonEntry":
        """Deserialize from JSON dict.

        Handles backward compatibility:
        - Old 'owner_user_id' key maps to 'created_by'
        - Old 'aliases' as list[str] converts to list[AliasEntry]
        - Old 'relationship' string converts to list[RelationshipClaim]
        - Old 'relation' key also supported
        """
        # Backward compat: owner_user_id -> created_by
        created_by = d.get("created_by") or d.get("owner_user_id", "")

        # Backward compat: aliases as plain strings -> AliasEntry
        raw_aliases = d.get("aliases") or []
        aliases: list[AliasEntry] = []
        for item in raw_aliases:
            if isinstance(item, str):
                aliases.append(AliasEntry(value=item))
            elif isinstance(item, dict):
                aliases.append(AliasEntry.from_dict(item))

        # Backward compat: relationship string -> RelationshipClaim list
        raw_relationships = d.get("relationships")
        relationships: list[RelationshipClaim] = []
        if raw_relationships and isinstance(raw_relationships, list):
            for item in raw_relationships:
                if isinstance(item, dict):
                    relationships.append(RelationshipClaim.from_dict(item))
        else:
            # Old format: single relationship string
            old_rel = d.get("relationship") or d.get("relation")
            if old_rel:
                relationships.append(RelationshipClaim(relationship=old_rel))

        return cls(
            id=d["id"],
            version=d.get("version", 1),
            created_by=created_by,
            name=d.get("name", ""),
            relationships=relationships,
            aliases=aliases,
            merged_into=d.get("merged_into"),
            created_at=_parse_datetime(d.get("created_at")),
            updated_at=_parse_datetime(d.get("updated_at")),
            metadata=d.get("metadata"),
        )


@dataclass
class PersonResolutionResult:
    """Result of person resolution."""

    person_id: str
    created: bool
    person_name: str
