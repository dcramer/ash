"""Entry types for JSONL session storage."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

# Session format version - increment when breaking format changes
SESSION_VERSION = "1"


def generate_id() -> str:
    """Generate a unique ID for entries."""
    return str(uuid.uuid4())


def now_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(UTC)


def session_key(
    provider: str,
    chat_id: str | None = None,
    user_id: str | None = None,
) -> str:
    """Generate a session directory key from components.

    Args:
        provider: Provider name (e.g., "cli", "telegram", "api").
        chat_id: Optional chat/conversation ID.
        user_id: Optional user ID (only used if no chat_id).

    Returns:
        Session key suitable for use as directory name.
    """
    parts = [_sanitize(provider)]
    if chat_id:
        parts.append(_sanitize(chat_id))
    elif user_id:
        parts.append(_sanitize(user_id))
    return "_".join(parts)


def _sanitize(s: str) -> str:
    """Sanitize a string for use in filesystem paths.

    Args:
        s: Input string.

    Returns:
        Sanitized string (alphanumeric + underscore, max 64 chars).
    """
    # Replace non-alphanumeric with underscore
    cleaned = re.sub(r"[^a-zA-Z0-9]", "_", s)
    # Collapse multiple underscores
    cleaned = re.sub(r"_+", "_", cleaned)
    # Strip leading/trailing underscores
    cleaned = cleaned.strip("_")
    # Limit length
    return cleaned[:64] if cleaned else "default"


@dataclass
class SessionHeader:
    """Session header entry - first line in context.jsonl."""

    id: str
    created_at: datetime
    provider: str
    user_id: str | None = None
    chat_id: str | None = None
    version: str = SESSION_VERSION
    type: Literal["session"] = "session"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "type": self.type,
            "version": self.version,
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "provider": self.provider,
            "user_id": self.user_id,
            "chat_id": self.chat_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionHeader:
        """Create from dict."""
        created_at = data["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return cls(
            id=data["id"],
            created_at=created_at,
            provider=data["provider"],
            user_id=data.get("user_id"),
            chat_id=data.get("chat_id"),
            version=data.get("version", SESSION_VERSION),
        )

    @classmethod
    def create(
        cls,
        provider: str,
        user_id: str | None = None,
        chat_id: str | None = None,
    ) -> SessionHeader:
        """Create a new session header."""
        return cls(
            id=generate_id(),
            created_at=now_utc(),
            provider=provider,
            user_id=user_id,
            chat_id=chat_id,
        )


@dataclass
class MessageEntry:
    """Message entry - user or assistant message."""

    id: str
    role: Literal["user", "assistant", "system"]
    content: str | list[dict[str, Any]]
    created_at: datetime
    token_count: int | None = None
    user_id: str | None = None
    metadata: dict[str, Any] | None = None  # For external_id, reply tracking, etc.
    type: Literal["message"] = "message"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for context.jsonl."""
        result: dict[str, Any] = {
            "type": self.type,
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "token_count": self.token_count,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def to_history_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for history.jsonl.

        Simplified format without type prefix.
        """
        result: dict[str, Any] = {
            "id": self.id,
            "role": self.role,
            "content": self._extract_text_content(),
            "created_at": self.created_at.isoformat(),
        }
        if self.user_id:
            result["user_id"] = self.user_id
        return result

    def _extract_text_content(self) -> str:
        """Extract text content from message.

        If content is a list of blocks, concatenate text blocks.
        """
        if isinstance(self.content, str):
            return self.content
        # Extract text from content blocks
        texts = []
        for block in self.content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts) if texts else ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageEntry:
        """Create from dict."""
        created_at = data["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return cls(
            id=data["id"],
            role=data["role"],
            content=data["content"],
            created_at=created_at,
            token_count=data.get("token_count"),
            user_id=data.get("user_id"),
            metadata=data.get("metadata"),
        )

    @classmethod
    def create(
        cls,
        role: Literal["user", "assistant", "system"],
        content: str | list[dict[str, Any]],
        token_count: int | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MessageEntry:
        """Create a new message entry."""
        return cls(
            id=generate_id(),
            role=role,
            content=content,
            created_at=now_utc(),
            token_count=token_count,
            user_id=user_id,
            metadata=metadata,
        )


@dataclass
class ToolUseEntry:
    """Tool use entry - request to execute a tool."""

    id: str
    message_id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "type": self.type,
            "id": self.id,
            "message_id": self.message_id,
            "name": self.name,
            "input": self.input,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolUseEntry:
        """Create from dict."""
        return cls(
            id=data["id"],
            message_id=data["message_id"],
            name=data["name"],
            input=data["input"],
        )

    @classmethod
    def create(
        cls,
        tool_use_id: str,
        message_id: str,
        name: str,
        input_data: dict[str, Any],
    ) -> ToolUseEntry:
        """Create a new tool use entry."""
        return cls(
            id=tool_use_id,
            message_id=message_id,
            name=name,
            input=input_data,
        )


@dataclass
class ToolResultEntry:
    """Tool result entry - result from tool execution."""

    tool_use_id: str
    output: str
    success: bool
    duration_ms: int | None = None
    type: Literal["tool_result"] = "tool_result"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "type": self.type,
            "tool_use_id": self.tool_use_id,
            "output": self.output,
            "success": self.success,
        }
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResultEntry:
        """Create from dict."""
        return cls(
            tool_use_id=data["tool_use_id"],
            output=data["output"],
            success=data["success"],
            duration_ms=data.get("duration_ms"),
        )

    @classmethod
    def create(
        cls,
        tool_use_id: str,
        output: str,
        success: bool,
        duration_ms: int | None = None,
    ) -> ToolResultEntry:
        """Create a new tool result entry."""
        return cls(
            tool_use_id=tool_use_id,
            output=output,
            success=success,
            duration_ms=duration_ms,
        )


@dataclass
class CompactionEntry:
    """Compaction entry - marks context window compression."""

    id: str
    summary: str
    tokens_before: int
    tokens_after: int
    first_kept_entry_id: str
    created_at: datetime = field(default_factory=now_utc)
    type: Literal["compaction"] = "compaction"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "type": self.type,
            "id": self.id,
            "summary": self.summary,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "first_kept_entry_id": self.first_kept_entry_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompactionEntry:
        """Create from dict."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = now_utc()
        return cls(
            id=data["id"],
            summary=data["summary"],
            tokens_before=data["tokens_before"],
            tokens_after=data["tokens_after"],
            first_kept_entry_id=data["first_kept_entry_id"],
            created_at=created_at,
        )

    @classmethod
    def create(
        cls,
        summary: str,
        tokens_before: int,
        tokens_after: int,
        first_kept_entry_id: str,
    ) -> CompactionEntry:
        """Create a new compaction entry."""
        return cls(
            id=generate_id(),
            summary=summary,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            first_kept_entry_id=first_kept_entry_id,
        )


# Union type for all entry types
Entry = SessionHeader | MessageEntry | ToolUseEntry | ToolResultEntry | CompactionEntry


def parse_entry(data: dict[str, Any]) -> Entry:
    """Parse a dict into the appropriate entry type.

    Args:
        data: Dict from JSON parsing.

    Returns:
        Typed entry object.

    Raises:
        ValueError: If entry type is unknown.
    """
    match data.get("type"):
        case "session":
            return SessionHeader.from_dict(data)
        case "message":
            return MessageEntry.from_dict(data)
        case "tool_use":
            return ToolUseEntry.from_dict(data)
        case "tool_result":
            return ToolResultEntry.from_dict(data)
        case "compaction":
            return CompactionEntry.from_dict(data)
        case unknown:
            raise ValueError(f"Unknown entry type: {unknown}")
