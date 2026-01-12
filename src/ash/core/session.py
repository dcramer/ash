"""Session management for conversation state."""

import json
from dataclasses import dataclass, field
from typing import Any

from ash.llm.types import ContentBlock, Message, Role, ToolResult, ToolUse
from ash.sessions.utils import (
    DEFAULT_RECENCY_WINDOW,
    content_block_from_dict,
    content_block_to_dict,
    prune_messages_to_budget,
)


@dataclass
class SessionState:
    """State for a conversation session."""

    session_id: str
    provider: str
    chat_id: str
    user_id: str
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Token tracking for smart pruning (populated from DB)
    _token_counts: list[int] = field(default_factory=list, repr=False)
    _message_ids: list[str] = field(default_factory=list, repr=False)

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

    def get_messages_for_llm(
        self,
        token_budget: int | None = None,
        recency_window: int = DEFAULT_RECENCY_WINDOW,
    ) -> list[Message]:
        """Get messages formatted for LLM, pruned to fit token budget.

        Args:
            token_budget: Maximum tokens for messages (None = no limit).
            recency_window: Always keep at least this many recent messages.

        Returns:
            List of messages within token budget.
        """
        if token_budget is None or not self.messages:
            return self.messages.copy()

        # Get token counts (use cached or estimate)
        token_counts = self._get_token_counts()

        # Use shared pruning logic
        pruned, _ = prune_messages_to_budget(
            self.messages,
            token_counts,
            token_budget,
            recency_window,
        )
        return pruned

    def _get_token_counts(self) -> list[int]:
        """Get token counts for all messages, estimating if not cached."""
        from ash.core.tokens import estimate_message_tokens

        if len(self._token_counts) == len(self.messages):
            return self._token_counts

        # Estimate missing counts
        counts: list[int] = []
        for i, msg in enumerate(self.messages):
            if i < len(self._token_counts):
                counts.append(self._token_counts[i])
            else:
                content = msg.content
                if isinstance(content, str):
                    counts.append(estimate_message_tokens(msg.role.value, content))
                else:
                    # Convert content blocks to dict format for estimation
                    blocks = [content_block_to_dict(b) for b in content]
                    counts.append(estimate_message_tokens(msg.role.value, blocks))

        return counts

    def set_token_counts(self, counts: list[int]) -> None:
        """Set cached token counts from DB.

        Args:
            counts: Token counts for messages (same order as messages).
        """
        self._token_counts = counts

    def set_message_ids(self, ids: list[str]) -> None:
        """Set message IDs (from DB) for deduplication.

        Args:
            ids: Message IDs corresponding to messages list.
        """
        self._message_ids = ids

    def get_recent_message_ids(self, recency_window: int) -> set[str]:
        """Get message IDs in the recency window.

        Args:
            recency_window: Number of recent messages.

        Returns:
            Set of message IDs.
        """
        if not self._message_ids:
            return set()
        start = max(0, len(self._message_ids) - recency_window)
        return set(self._message_ids[start:])

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

    def has_incomplete_tool_use(self) -> bool:
        """Check if session has tool_use without matching tool_result.

        This can happen if a message was interrupted during tool execution.

        Returns:
            True if there are incomplete tool uses.
        """
        return len(self.get_pending_tool_uses()) > 0

    def repair_incomplete_tool_use(self) -> bool:
        """Repair session state with incomplete tool_use blocks.

        If the last assistant message has tool_use without tool_result,
        add error tool_results to make the session valid.

        Returns:
            True if repairs were made, False otherwise.
        """
        pending = self.get_pending_tool_uses()
        if not pending:
            return False

        # Add error tool_results for each pending tool_use
        for tool_use in pending:
            self.add_tool_result(
                tool_use_id=tool_use.id,
                content="[Tool execution was interrupted]",
                is_error=True,
            )

        return True

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
            content = [content_block_to_dict(block) for block in message.content]

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
            content = [
                block
                for block_data in raw_content
                if (block := content_block_from_dict(block_data)) is not None
            ]

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
