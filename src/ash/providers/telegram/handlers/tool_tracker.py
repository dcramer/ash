"""Tool tracking and thinking message management for Telegram.

This module provides:
- ToolTracker: Tracks tool calls and manages the "Thinking..." message
- ProgressMessageTool: Per-run send_message tool for progress updates
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ash.providers.base import OutgoingMessage
from ash.providers.telegram.handlers.utils import (
    MAX_MESSAGE_LENGTH,
    escape_markdown_v2,
    format_tool_brief,
)

if TYPE_CHECKING:
    from ash.agents import AgentRegistry
    from ash.config import AshConfig
    from ash.providers.telegram.provider import TelegramProvider
    from ash.skills import SkillRegistry


class ProgressMessageTool:
    """Per-run send_message tool that appends to the thinking message.

    This tool replaces the default send_message tool during agent execution,
    so progress updates appear in the consolidated thinking message instead
    of being sent as separate replies.
    """

    def __init__(self, tracker: ToolTracker) -> None:
        self._tracker = tracker

    @property
    def name(self) -> str:
        return "send_message"

    @property
    def description(self) -> str:
        return (
            "Send a progress update to the user. "
            "Use for status updates or intermediate results. "
            "The message appears in the current response thread."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The progress message to display",
                },
            },
            "required": ["message"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: Any,  # ToolContext, but we don't need to type it strictly
    ) -> Any:
        from ash.tools.base import ToolResult

        message = input_data.get("message", "").strip()
        if not message:
            return ToolResult.error("Message cannot be empty")

        self._tracker.add_progress_message(message)
        await self._tracker.update_display()
        return ToolResult.success("Progress message added")

    def to_definition(self) -> dict[str, Any]:
        """Convert to LLM tool definition format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


THINKING_STATUS = "_Thinking\\.\\.\\._"


class ToolTracker:
    """Tracks tool calls and manages thinking message updates.

    Consolidates all progress into a single message that gets edited:
    - Progress messages at the top (via add_progress_message)
    - "Thinking..." status always at the bottom
    - Final response replaces "Thinking..." with response content
    """

    def __init__(
        self,
        provider: TelegramProvider,
        chat_id: str,
        reply_to: str,
        config: AshConfig | None = None,
        agent_registry: AgentRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
    ):
        self._provider = provider
        self._chat_id = chat_id
        self._reply_to = reply_to
        self._config = config
        self._agent_registry = agent_registry
        self._skill_registry = skill_registry
        self.thinking_msg_id: str | None = None
        self.tool_count: int = 0
        self.progress_messages: list[str] = []

    def _build_display_message(self, *, include_thinking: bool = True) -> str:
        """Build the consolidated message with progress on top and status on bottom.

        Args:
            include_thinking: Whether to include the "Thinking..." line at the bottom.

        Returns:
            Message content, truncated to fit Telegram's limit.
        """
        parts: list[str] = []

        if self.progress_messages:
            escaped = [escape_markdown_v2(m) for m in self.progress_messages]
            parts.extend(escaped)

        if include_thinking:
            if parts:
                parts.append("")  # Blank line before thinking
            parts.append(THINKING_STATUS)

        message = "\n".join(parts)

        # If under limit, return as-is
        if len(message) <= MAX_MESSAGE_LENGTH:
            return message

        # Truncate oldest progress messages until it fits
        truncated_progress = self.progress_messages.copy()
        truncation_notice = escape_markdown_v2("[...earlier messages truncated...]")

        while truncated_progress and len(message) > MAX_MESSAGE_LENGTH:
            truncated_progress.pop(0)
            parts = []
            if truncated_progress:
                parts.append(truncation_notice)
                parts.extend(escape_markdown_v2(m) for m in truncated_progress)
            if include_thinking:
                if parts:
                    parts.append("")
                parts.append(THINKING_STATUS)
            message = "\n".join(parts)

        return message

    async def on_tool_start(self, tool_name: str, tool_input: dict[str, Any]) -> None:
        """Record a tool call and update the thinking message."""
        # Validate tool call (for logging purposes, but don't block)
        format_tool_brief(
            tool_name,
            tool_input,
            config=self._config,
            agent_registry=self._agent_registry,
            skill_registry=self._skill_registry,
        )

        self.tool_count += 1
        display_message = self._build_display_message()

        if self.thinking_msg_id is None:
            self.thinking_msg_id = await self._provider.send(
                OutgoingMessage(
                    chat_id=self._chat_id,
                    text=display_message,
                    reply_to_message_id=self._reply_to,
                    parse_mode="markdown_v2",
                )
            )
        else:
            await self._provider.edit(
                self._chat_id,
                self.thinking_msg_id,
                display_message,
                parse_mode="markdown_v2",
            )

    def add_progress_message(self, message: str) -> None:
        """Add a progress message to be displayed."""
        self.progress_messages.append(message)

    async def update_display(self) -> None:
        """Update the thinking message with current progress."""
        display_message = self._build_display_message()

        if self.thinking_msg_id is None:
            self.thinking_msg_id = await self._provider.send(
                OutgoingMessage(
                    chat_id=self._chat_id,
                    text=display_message,
                    reply_to_message_id=self._reply_to,
                    parse_mode="markdown_v2",
                )
            )
        else:
            await self._provider.edit(
                self._chat_id,
                self.thinking_msg_id,
                display_message,
                parse_mode="markdown_v2",
            )

    async def finalize_response(self, response_content: str) -> str:
        """Build final content and edit/send the response, returning message ID.

        Final content = progress messages + response content (no "Thinking...", no stats).
        """
        if self.progress_messages:
            parts = (
                self.progress_messages + ["", response_content]
                if response_content
                else list(self.progress_messages)
            )
            final_content = "\n".join(parts)
        else:
            final_content = response_content

        if self.thinking_msg_id:
            await self._provider.edit(
                self._chat_id, self.thinking_msg_id, final_content
            )
            return self.thinking_msg_id

        return await self._provider.send(
            OutgoingMessage(
                chat_id=self._chat_id,
                text=response_content,
                reply_to_message_id=self._reply_to,
            )
        )
