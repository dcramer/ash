"""Session management for conversation state."""

import json
from dataclasses import dataclass, field
from typing import Any

from ash.llm.types import ContentBlock, Message, Role, TextContent, ToolResult, ToolUse


@dataclass
class SessionState:
    """State for a conversation session."""

    session_id: str
    provider: str
    chat_id: str
    user_id: str
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_user_message(self, content: str) -> Message:
        """Add a user message to the session.

        Args:
            content: Message content.

        Returns:
            Created message.
        """
        message = Message(role=Role.USER, content=content)
        self.messages.append(message)
        return message

    def add_assistant_message(self, content: str | list[ContentBlock]) -> Message:
        """Add an assistant message to the session.

        Args:
            content: Message content or content blocks.

        Returns:
            Created message.
        """
        message = Message(role=Role.ASSISTANT, content=content)
        self.messages.append(message)
        return message

    def add_tool_result(
        self,
        tool_use_id: str,
        content: str,
        is_error: bool = False,
    ) -> Message:
        """Add a tool result message to the session.

        Args:
            tool_use_id: ID of the tool use this is a result for.
            content: Result content.
            is_error: Whether this is an error result.

        Returns:
            Created message.
        """
        result = ToolResult(
            tool_use_id=tool_use_id,
            content=content,
            is_error=is_error,
        )
        message = Message(role=Role.USER, content=[result])
        self.messages.append(message)
        return message

    def get_messages_for_llm(self) -> list[Message]:
        """Get messages formatted for LLM.

        Returns:
            List of messages.
        """
        return self.messages.copy()

    def get_pending_tool_uses(self) -> list[ToolUse]:
        """Get tool uses from the last assistant message that need results.

        Returns:
            List of tool uses.
        """
        if not self.messages:
            return []

        last_message = self.messages[-1]
        if last_message.role != Role.ASSISTANT:
            return []

        if isinstance(last_message.content, str):
            return []

        return [block for block in last_message.content if isinstance(block, ToolUse)]

    def get_last_text_response(self) -> str | None:
        """Get the text content of the last assistant message.

        Returns:
            Text content or None.
        """
        for message in reversed(self.messages):
            if message.role == Role.ASSISTANT:
                return message.get_text()
        return None

    def clear_messages(self) -> None:
        """Clear all messages from the session."""
        self.messages.clear()

    def to_dict(self) -> dict[str, Any]:
        """Convert session state to dict for storage.

        Returns:
            Dict representation.
        """
        return {
            "session_id": self.session_id,
            "provider": self.provider,
            "chat_id": self.chat_id,
            "user_id": self.user_id,
            "messages": [self._message_to_dict(m) for m in self.messages],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionState":
        """Create session state from dict.

        Args:
            data: Dict representation.

        Returns:
            Session state.
        """
        messages = [cls._message_from_dict(m) for m in data.get("messages", [])]
        return cls(
            session_id=data["session_id"],
            provider=data["provider"],
            chat_id=data["chat_id"],
            user_id=data["user_id"],
            messages=messages,
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def _message_to_dict(message: Message) -> dict[str, Any]:
        """Convert message to dict.

        Args:
            message: Message to convert.

        Returns:
            Dict representation.
        """
        if isinstance(message.content, str):
            content = message.content
        else:
            content = []
            for block in message.content:
                if isinstance(block, TextContent):
                    content.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUse):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
                elif isinstance(block, ToolResult):
                    content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.tool_use_id,
                            "content": block.content,
                            "is_error": block.is_error,
                        }
                    )

        return {
            "role": message.role.value,
            "content": content,
        }

    @staticmethod
    def _message_from_dict(data: dict[str, Any]) -> Message:
        """Create message from dict.

        Args:
            data: Dict representation.

        Returns:
            Message.
        """
        role = Role(data["role"])
        raw_content = data["content"]

        if isinstance(raw_content, str):
            content: str | list[ContentBlock] = raw_content
        else:
            content = []
            for block in raw_content:
                block_type = block.get("type")
                if block_type == "text":
                    content.append(TextContent(text=block["text"]))
                elif block_type == "tool_use":
                    content.append(
                        ToolUse(
                            id=block["id"],
                            name=block["name"],
                            input=block["input"],
                        )
                    )
                elif block_type == "tool_result":
                    content.append(
                        ToolResult(
                            tool_use_id=block["tool_use_id"],
                            content=block["content"],
                            is_error=block.get("is_error", False),
                        )
                    )

        return Message(role=role, content=content)

    def to_json(self) -> str:
        """Serialize session state to JSON.

        Returns:
            JSON string.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "SessionState":
        """Create session state from JSON.

        Args:
            json_str: JSON string.

        Returns:
            Session state.
        """
        return cls.from_dict(json.loads(json_str))
