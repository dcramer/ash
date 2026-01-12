"""Shared utilities for session management."""

from __future__ import annotations

import logging
from typing import Any

from ash.llm.types import (
    ContentBlock,
    Message,
    Role,
    TextContent,
    ToolResult,
    ToolUse,
)

logger = logging.getLogger(__name__)


def content_block_to_dict(block: ContentBlock) -> dict[str, Any]:
    """Convert a ContentBlock to dict for storage.

    Args:
        block: Content block to convert.

    Returns:
        Dict representation.
    """
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    elif isinstance(block, ToolUse):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    elif isinstance(block, ToolResult):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    return {}


def validate_tool_pairs(
    messages: list[Message],
    message_ids: list[str] | None = None,
) -> tuple[list[Message], list[str]]:
    """Validate and fix tool_use/tool_result pairs.

    Handles two cases:
    1. Orphaned tool_results (tool_result without tool_use) - removed
    2. Orphaned tool_uses (tool_use without tool_result) - synthetic result inserted

    Args:
        messages: List of messages.
        message_ids: Corresponding message IDs (optional, defaults to empty strings).

    Returns:
        Tuple of (validated messages, validated IDs).
    """
    if not messages:
        return messages, message_ids or []

    # Use empty strings for IDs if not provided
    if message_ids is None:
        message_ids = [""] * len(messages)

    result_msgs: list[Message] = []
    result_ids: list[str] = []

    # Track tool_use IDs and which have results
    pending_tool_uses: list[ToolUse] = []  # Tool uses awaiting results
    seen_tool_use_ids: set[str] = set()

    for msg, msg_id in zip(messages, message_ids, strict=False):
        # Assistant messages: collect tool_uses
        if msg.role == Role.ASSISTANT and isinstance(msg.content, list):
            # First, check if we have pending tool_uses from a previous assistant
            # message that never got results - insert synthetic results
            if pending_tool_uses:
                synthetic_results: list[ContentBlock] = [
                    ToolResult(
                        tool_use_id=tu.id,
                        content="[No result - execution was interrupted]",
                        is_error=True,
                    )
                    for tu in pending_tool_uses
                ]
                result_msgs.append(Message(role=Role.USER, content=synthetic_results))
                result_ids.append("")  # Synthetic message has no ID
                logger.warning(
                    "Inserted %d synthetic tool_result(s) for orphaned tool_use(s)",
                    len(pending_tool_uses),
                )
                pending_tool_uses = []

            # Collect new tool_uses from this message
            for block in msg.content:
                if isinstance(block, ToolUse):
                    seen_tool_use_ids.add(block.id)
                    pending_tool_uses.append(block)

            result_msgs.append(msg)
            result_ids.append(msg_id)

        # User messages with tool_results: validate and mark as satisfied
        elif msg.role == Role.USER and isinstance(msg.content, list):
            has_tool_results = any(
                isinstance(block, ToolResult) for block in msg.content
            )

            if has_tool_results:
                # Filter to only tool_results with matching tool_uses
                valid_content: list[ContentBlock] = []
                for block in msg.content:
                    if isinstance(block, ToolResult):
                        if block.tool_use_id in seen_tool_use_ids:
                            valid_content.append(block)
                            # Remove from pending - this tool_use is satisfied
                            pending_tool_uses = [
                                tu
                                for tu in pending_tool_uses
                                if tu.id != block.tool_use_id
                            ]
                        else:
                            logger.warning(
                                "Removing orphaned tool_result: %s",
                                block.tool_use_id,
                            )
                    else:
                        valid_content.append(block)

                # Only add message if it still has content
                if valid_content:
                    result_msgs.append(Message(role=msg.role, content=valid_content))
                    result_ids.append(msg_id)
            else:
                result_msgs.append(msg)
                result_ids.append(msg_id)
        else:
            result_msgs.append(msg)
            result_ids.append(msg_id)

    # Handle any remaining pending tool_uses at the end
    if pending_tool_uses:
        synthetic_results: list[ContentBlock] = [
            ToolResult(
                tool_use_id=tu.id,
                content="[No result - execution was interrupted]",
                is_error=True,
            )
            for tu in pending_tool_uses
        ]
        result_msgs.append(Message(role=Role.USER, content=synthetic_results))
        result_ids.append("")
        logger.warning(
            "Inserted %d synthetic tool_result(s) for orphaned tool_use(s) at end",
            len(pending_tool_uses),
        )

    return result_msgs, result_ids
