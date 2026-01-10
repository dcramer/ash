"""LLM message types and data structures."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Message role."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ContentBlockType(str, Enum):
    """Content block type."""

    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class StreamEventType(str, Enum):
    """Stream event type."""

    TEXT_DELTA = "text_delta"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_DELTA = "tool_use_delta"
    TOOL_USE_END = "tool_use_end"
    MESSAGE_START = "message_start"
    MESSAGE_END = "message_end"
    ERROR = "error"


@dataclass
class TextContent:
    """Text content block."""

    text: str
    type: ContentBlockType = ContentBlockType.TEXT


@dataclass
class ToolUse:
    """Tool use request from LLM."""

    id: str
    name: str
    input: dict[str, Any]
    type: ContentBlockType = ContentBlockType.TOOL_USE


@dataclass
class ToolResult:
    """Tool execution result to send back to LLM."""

    tool_use_id: str
    content: str
    is_error: bool = False
    type: ContentBlockType = ContentBlockType.TOOL_RESULT


ContentBlock = TextContent | ToolUse | ToolResult


@dataclass
class Message:
    """A message in the conversation."""

    role: Role
    content: str | list[ContentBlock]

    def get_text(self) -> str:
        """Extract text content from message."""
        if isinstance(self.content, str):
            return self.content
        texts = [block.text for block in self.content if isinstance(block, TextContent)]
        return "\n".join(texts)

    def get_tool_uses(self) -> list[ToolUse]:
        """Extract tool use requests from message."""
        if isinstance(self.content, str):
            return []
        return [block for block in self.content if isinstance(block, ToolUse)]


@dataclass
class ToolDefinition:
    """Tool definition for LLM."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class StreamChunk:
    """A chunk from streaming response."""

    type: StreamEventType
    content: str | dict[str, Any] | None = None
    tool_use_id: str | None = None
    tool_name: str | None = None


@dataclass
class Usage:
    """Token usage information."""

    input_tokens: int
    output_tokens: int


@dataclass
class CompletionResponse:
    """Full completion response."""

    message: Message
    usage: Usage | None = None
    stop_reason: str | None = None
    model: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
