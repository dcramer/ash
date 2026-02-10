"""Entry types for JSONL session storage."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SESSION_VERSION = "1"


def generate_id() -> str:
    return str(uuid.uuid4())


def now_utc() -> datetime:
    return datetime.now(UTC)


def _parse_datetime(value: str | datetime | None) -> datetime:
    if value is None:
        return now_utc()
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def session_key(
    provider: str,
    chat_id: str | None = None,
    user_id: str | None = None,
    thread_id: str | None = None,
) -> str:
    parts = [_sanitize(provider)]
    if chat_id:
        parts.append(_sanitize(chat_id))
        if thread_id:
            parts.append(_sanitize(thread_id))
    elif user_id:
        parts.append(_sanitize(user_id))
    return "_".join(parts)


def _sanitize(s: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\-]", "_", s)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned[:64] if cleaned else "default"


@dataclass
class SessionHeader:
    id: str
    created_at: datetime
    provider: str
    user_id: str | None = None
    chat_id: str | None = None
    version: str = SESSION_VERSION
    type: Literal["session"] = "session"

    def to_dict(self) -> dict[str, Any]:
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
        return cls(
            id=data["id"],
            created_at=_parse_datetime(data["created_at"]),
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
        return cls(
            id=generate_id(),
            created_at=now_utc(),
            provider=provider,
            user_id=user_id,
            chat_id=chat_id,
        )


@dataclass
class AgentSessionEntry:
    """Entry marking the start of a subagent session.

    Links subagent execution to the parent session's tool_use that invoked it.
    All subsequent entries with matching agent_session_id belong to this subagent.
    """

    id: str
    parent_tool_use_id: str  # Links to the tool_use that invoked this agent
    agent_type: Literal["skill", "agent"]  # Type of subagent
    agent_name: str  # Name of the skill or agent
    created_at: datetime
    type: Literal["agent_session"] = "agent_session"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "parent_tool_use_id": self.parent_tool_use_id,
            "agent_type": self.agent_type,
            "agent_name": self.agent_name,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSessionEntry:
        return cls(
            id=data["id"],
            parent_tool_use_id=data["parent_tool_use_id"],
            agent_type=data["agent_type"],
            agent_name=data["agent_name"],
            created_at=_parse_datetime(data["created_at"]),
        )

    @classmethod
    def create(
        cls,
        parent_tool_use_id: str,
        agent_type: Literal["skill", "agent"],
        agent_name: str,
    ) -> AgentSessionEntry:
        return cls(
            id=generate_id(),
            parent_tool_use_id=parent_tool_use_id,
            agent_type=agent_type,
            agent_name=agent_name,
            created_at=now_utc(),
        )


@dataclass
class MessageEntry:
    id: str
    role: Literal["user", "assistant", "system"]
    content: str | list[dict[str, Any]]
    created_at: datetime
    token_count: int | None = None
    user_id: str | None = None
    username: str | None = None
    display_name: str | None = None
    metadata: dict[str, Any] | None = None
    agent_session_id: str | None = None  # Links to AgentSessionEntry for subagent msgs
    type: Literal["message"] = "message"

    def to_dict(self) -> dict[str, Any]:
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
        if self.agent_session_id:
            result["agent_session_id"] = self.agent_session_id
        return result

    def to_history_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "role": self.role,
            "content": self._extract_text_content(),
            "created_at": self.created_at.isoformat(),
        }
        if self.user_id:
            result["user_id"] = self.user_id
        if self.username:
            result["username"] = self.username
        if self.display_name:
            result["display_name"] = self.display_name
        return result

    def _extract_text_content(self) -> str:
        if isinstance(self.content, str):
            return self.content
        texts = [
            block.get("text", "")
            for block in self.content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(texts)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageEntry:
        return cls(
            id=data["id"],
            role=data["role"],
            content=data["content"],
            created_at=_parse_datetime(data["created_at"]),
            token_count=data.get("token_count"),
            user_id=data.get("user_id"),
            username=data.get("username"),
            display_name=data.get("display_name"),
            metadata=data.get("metadata"),
            agent_session_id=data.get("agent_session_id"),
        )

    @classmethod
    def create(
        cls,
        role: Literal["user", "assistant", "system"],
        content: str | list[dict[str, Any]],
        token_count: int | None = None,
        user_id: str | None = None,
        username: str | None = None,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        agent_session_id: str | None = None,
    ) -> MessageEntry:
        return cls(
            id=generate_id(),
            role=role,
            content=content,
            created_at=now_utc(),
            token_count=token_count,
            user_id=user_id,
            username=username,
            display_name=display_name,
            metadata=metadata,
            agent_session_id=agent_session_id,
        )


@dataclass
class ToolUseEntry:
    id: str
    message_id: str
    name: str
    input: dict[str, Any]
    agent_session_id: str | None = None  # Links to AgentSessionEntry for subagent calls
    type: Literal["tool_use"] = "tool_use"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "id": self.id,
            "message_id": self.message_id,
            "name": self.name,
            "input": self.input,
        }
        if self.agent_session_id:
            result["agent_session_id"] = self.agent_session_id
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolUseEntry:
        return cls(
            id=data["id"],
            message_id=data["message_id"],
            name=data["name"],
            input=data["input"],
            agent_session_id=data.get("agent_session_id"),
        )

    @classmethod
    def create(
        cls,
        tool_use_id: str,
        message_id: str,
        name: str,
        input_data: dict[str, Any],
        agent_session_id: str | None = None,
    ) -> ToolUseEntry:
        return cls(
            id=tool_use_id,
            message_id=message_id,
            name=name,
            input=input_data,
            agent_session_id=agent_session_id,
        )


@dataclass
class ToolResultEntry:
    tool_use_id: str
    output: str
    success: bool
    duration_ms: int | None = None
    metadata: dict[str, Any] | None = None
    agent_session_id: str | None = None  # Links to AgentSessionEntry for subagent calls
    type: Literal["tool_result"] = "tool_result"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "tool_use_id": self.tool_use_id,
            "output": self.output,
            "success": self.success,
        }
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.metadata is not None:
            result["metadata"] = self.metadata
        if self.agent_session_id:
            result["agent_session_id"] = self.agent_session_id
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResultEntry:
        return cls(
            tool_use_id=data["tool_use_id"],
            output=data["output"],
            success=data["success"],
            duration_ms=data.get("duration_ms"),
            metadata=data.get("metadata"),
            agent_session_id=data.get("agent_session_id"),
        )

    @classmethod
    def create(
        cls,
        tool_use_id: str,
        output: str,
        success: bool,
        duration_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
        agent_session_id: str | None = None,
    ) -> ToolResultEntry:
        return cls(
            tool_use_id=tool_use_id,
            output=output,
            success=success,
            duration_ms=duration_ms,
            metadata=metadata,
            agent_session_id=agent_session_id,
        )


@dataclass
class CompactionEntry:
    id: str
    summary: str
    tokens_before: int
    tokens_after: int
    first_kept_entry_id: str
    created_at: datetime = field(default_factory=now_utc)
    type: Literal["compaction"] = "compaction"

    def to_dict(self) -> dict[str, Any]:
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
        return cls(
            id=data["id"],
            summary=data["summary"],
            tokens_before=data["tokens_before"],
            tokens_after=data["tokens_after"],
            first_kept_entry_id=data["first_kept_entry_id"],
            created_at=_parse_datetime(data.get("created_at")),
        )

    @classmethod
    def create(
        cls,
        summary: str,
        tokens_before: int,
        tokens_after: int,
        first_kept_entry_id: str,
    ) -> CompactionEntry:
        return cls(
            id=generate_id(),
            summary=summary,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            first_kept_entry_id=first_kept_entry_id,
        )


Entry = (
    SessionHeader
    | AgentSessionEntry
    | MessageEntry
    | ToolUseEntry
    | ToolResultEntry
    | CompactionEntry
)

_ENTRY_PARSERS: dict[str, type[Entry]] = {
    "session": SessionHeader,
    "agent_session": AgentSessionEntry,
    "message": MessageEntry,
    "tool_use": ToolUseEntry,
    "tool_result": ToolResultEntry,
    "compaction": CompactionEntry,
}


def parse_entry(data: dict[str, Any]) -> Entry:
    entry_type = data.get("type")
    parser = _ENTRY_PARSERS.get(entry_type)  # type: ignore[arg-type]
    if parser is None:
        raise ValueError(f"Unknown entry type: {entry_type}")
    return parser.from_dict(data)


class SessionState(BaseModel):
    """Session metadata stored in state.json for easy lookup."""

    provider: str
    chat_id: str | None = None
    user_id: str | None = None
    thread_id: str | None = None
    created_at: datetime = Field(default_factory=now_utc)
