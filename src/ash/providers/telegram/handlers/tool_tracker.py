"""Tool tracking and thinking message management for Telegram.

This module provides:
- ToolTracker: Tracks tool calls and manages the "Thinking..." message
- ProgressMessageTool: Per-run send_message tool for progress updates
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ash.providers.base import OutgoingMessage
from ash.providers.telegram.handlers.utils import (
    MAX_MESSAGE_LENGTH,
    escape_markdown_v2,
    format_thinking_status,
    format_tool_brief,
    format_tool_summary,
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


class ToolTracker:
    """Tracks tool calls and manages thinking message updates.

    Consolidates all progress into a single message that gets edited:
    - Status line: "Thinking... (N tool calls)" or "Made N tool calls in Xs"
    - Progress messages: Appended via add_progress_message()
    - Final response: Appended at the end
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
        self.start_time: float | None = None

    def _build_display_message(
        self, status: str, final_content: str = "", *, escape_progress: bool = True
    ) -> str:
        """Build the consolidated message, truncating if needed.

        Args:
            status: The status line (pre-escaped for MarkdownV2 if used with that mode)
            final_content: Optional final response content (NOT escaped)
            escape_progress: Whether to escape progress messages for MarkdownV2.
                Set to True when the message will be sent with parse_mode="markdown_v2".

        Returns:
            Message content, truncated to fit Telegram's limit.

        Note:
            This method combines MarkdownV2-escaped status with progress messages.
            Progress messages are escaped when escape_progress=True to prevent
            special characters from breaking the MarkdownV2 parsing.
        """
        parts = [status]

        if self.progress_messages:
            parts.append("")  # Blank line after status
            if escape_progress:
                escaped = [escape_markdown_v2(m) for m in self.progress_messages]
                parts.extend(escaped)
            else:
                parts.extend(self.progress_messages)

        if final_content:
            parts.append("")  # Blank line before final content
            parts.append(final_content)

        message = "\n".join(parts)

        # If under limit, return as-is
        if len(message) <= MAX_MESSAGE_LENGTH:
            return message

        # Truncate oldest progress messages until it fits
        # Keep status + final content, drop progress messages from the start
        truncated_progress = self.progress_messages.copy()
        truncation_notice = (
            escape_markdown_v2("[...earlier messages truncated...]")
            if escape_progress
            else "[...earlier messages truncated...]"
        )

        while truncated_progress and len(message) > MAX_MESSAGE_LENGTH:
            truncated_progress.pop(0)
            parts = [status]
            if truncated_progress:
                parts.append("")
                parts.append(truncation_notice)
                if escape_progress:
                    parts.extend(escape_markdown_v2(m) for m in truncated_progress)
                else:
                    parts.extend(truncated_progress)
            if final_content:
                parts.append("")
                parts.append(final_content)
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

        if self.start_time is None:
            self.start_time = time.monotonic()

        self.tool_count += 1
        status = format_thinking_status(self.tool_count)
        display_message = self._build_display_message(status)

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
        if self.thinking_msg_id is None:
            # Create initial message if none exists
            status = format_thinking_status(self.tool_count)
            display_message = self._build_display_message(status)
            self.thinking_msg_id = await self._provider.send(
                OutgoingMessage(
                    chat_id=self._chat_id,
                    text=display_message,
                    reply_to_message_id=self._reply_to,
                    parse_mode="markdown_v2",
                )
            )
        else:
            status = format_thinking_status(self.tool_count)
            display_message = self._build_display_message(status)
            await self._provider.edit(
                self._chat_id,
                self.thinking_msg_id,
                display_message,
                parse_mode="markdown_v2",
            )

    def get_summary_prefix(self) -> str:
        """Get the summary line for the final message."""
        if self.tool_count > 0 and self.start_time:
            elapsed = time.monotonic() - self.start_time
            return format_tool_summary(self.tool_count, elapsed)
        return ""

    async def finalize_response(self, response_content: str) -> str:
        """Send or edit the final response, returning the message ID.

        The final response is edited with regular MARKDOWN mode (not MarkdownV2),
        so progress messages are NOT escaped. This allows the response content
        to use standard markdown formatting.
        """
        summary = self.get_summary_prefix()
        final_content = (
            self._build_display_message(
                summary, response_content, escape_progress=False
            )
            if summary
            else response_content
        )

        # Include progress messages in final content if we have them but no summary
        if not summary and self.progress_messages:
            parts = self.progress_messages + (
                ["", response_content] if response_content else []
            )
            final_content = "\n".join(parts)

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
