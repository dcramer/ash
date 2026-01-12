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

# Default number of recent messages to always include in context
DEFAULT_RECENCY_WINDOW = 10


def fit_messages_to_budget(
    messages: list[Message],
    token_counts: list[int],
    budget: int,
    message_ids: list[str] | None = None,
) -> tuple[list[Message], list[str]]:
    """Fit messages to token budget, keeping most recent.

    Iterates backward from most recent message, accumulating messages
    that fit within the budget.

    Args:
        messages: Messages to fit.
        token_counts: Token counts per message (same length as messages).
        budget: Maximum tokens to include.
        message_ids: Optional message IDs (defaults to empty strings).

    Returns:
        Tuple of (pruned messages, pruned IDs).
    """
    if message_ids is None:
        message_ids = [""] * len(messages)

    result_msgs: list[Message] = []
    result_ids: list[str] = []
    remaining = budget

    for msg, msg_id, tokens in zip(
        reversed(messages),
        reversed(message_ids),
        reversed(token_counts),
        strict=False,
    ):
        if tokens <= remaining:
            result_msgs.insert(0, msg)
            result_ids.insert(0, msg_id)
            remaining -= tokens
        else:
            break

    return result_msgs, result_ids


def prune_messages_to_budget(
    messages: list[Message],
    token_counts: list[int],
    token_budget: int | None,
    recency_window: int = DEFAULT_RECENCY_WINDOW,
    message_ids: list[str] | None = None,
) -> tuple[list[Message], list[str]]:
    """Prune messages to fit within token budget while preserving recency window.

    Algorithm:
    1. Always keep at least `recency_window` recent messages
    2. If recency window exceeds budget, fit what we can from it
    3. Otherwise, add older messages backward until budget exhausted
    4. Validate tool_use/tool_result pairs after pruning

    Args:
        messages: All messages.
        token_counts: Token counts per message (same length as messages).
        token_budget: Maximum tokens (None = no limit).
        recency_window: Always keep at least this many recent messages.
        message_ids: Optional message IDs (defaults to empty strings).

    Returns:
        Tuple of (pruned messages, pruned IDs).
    """
    if message_ids is None:
        message_ids = [""] * len(messages)

    if token_budget is None or not messages:
        return messages.copy(), list(message_ids)

    n_messages = len(messages)
    recency_start = max(0, n_messages - recency_window)

    # Calculate tokens in recency window
    recency_tokens = sum(token_counts[recency_start:])

    if recency_tokens >= token_budget:
        # Even recency window exceeds budget - fit what we can
        pruned_msgs, pruned_ids = fit_messages_to_budget(
            messages[recency_start:],
            token_counts[recency_start:],
            token_budget,
            message_ids[recency_start:],
        )
        return validate_tool_pairs(pruned_msgs, pruned_ids)

    # Budget remaining for older messages
    remaining_budget = token_budget - recency_tokens

    # Add older messages from most recent backward
    older_messages = messages[:recency_start]
    older_ids = message_ids[:recency_start]
    older_tokens = token_counts[:recency_start]

    included_msgs: list[Message] = []
    included_ids: list[str] = []

    for msg, msg_id, tokens in zip(
        reversed(older_messages),
        reversed(older_ids),
        reversed(older_tokens),
        strict=False,
    ):
        if tokens <= remaining_budget:
            included_msgs.insert(0, msg)
            included_ids.insert(0, msg_id)
            remaining_budget -= tokens
        else:
            break

    # Combine older + recent
    combined_msgs = included_msgs + messages[recency_start:]
    combined_ids = included_ids + message_ids[recency_start:]

    # Validate tool pairs - pruning may have orphaned some
    return validate_tool_pairs(combined_msgs, combined_ids)


def content_block_to_dict(block: ContentBlock) -> dict[str, Any]:
    """Convert a ContentBlock to dict for storage.

    Args:
        block: Content block to convert.

    Returns:
        Dict representation.
    """
    match block:
        case TextContent():
            return {"type": "text", "text": block.text}
        case ToolUse():
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }
        case ToolResult():
            return {
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
                "content": block.content,
                "is_error": block.is_error,
            }
        case _:
            return {}


def content_block_from_dict(data: dict[str, Any]) -> ContentBlock | None:
    """Create a ContentBlock from dict representation.

    Args:
        data: Dict with block data (must have "type" key).

    Returns:
        ContentBlock instance or None if type is unrecognized.
    """
    match data.get("type"):
        case "text":
            return TextContent(text=data["text"])
        case "tool_use":
            return ToolUse(
                id=data["id"],
                name=data["name"],
                input=data["input"],
            )
        case "tool_result":
            return ToolResult(
                tool_use_id=data["tool_use_id"],
                content=data["content"],
                is_error=data.get("is_error", False),
            )
        case _:
            return None


def _make_synthetic_results(tool_uses: list[ToolUse]) -> Message:
    """Create a synthetic user message with error results for orphaned tool uses."""
    results: list[ContentBlock] = [
        ToolResult(
            tool_use_id=tu.id,
            content="[No result - execution was interrupted]",
            is_error=True,
        )
        for tu in tool_uses
    ]
    return Message(role=Role.USER, content=results)


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

    if message_ids is None:
        message_ids = [""] * len(messages)

    result_msgs: list[Message] = []
    result_ids: list[str] = []
    pending_tool_uses: list[ToolUse] = []
    seen_tool_use_ids: set[str] = set()

    def flush_pending_tool_uses() -> None:
        """Insert synthetic results for any pending tool uses."""
        if not pending_tool_uses:
            return
        result_msgs.append(_make_synthetic_results(pending_tool_uses))
        result_ids.append("")
        logger.warning(
            "Inserted %d synthetic tool_result(s) for orphaned tool_use(s)",
            len(pending_tool_uses),
        )
        pending_tool_uses.clear()

    for msg, msg_id in zip(messages, message_ids, strict=False):
        # Assistant messages: collect tool_uses
        if msg.role == Role.ASSISTANT and isinstance(msg.content, list):
            flush_pending_tool_uses()

            for block in msg.content:
                if isinstance(block, ToolUse):
                    seen_tool_use_ids.add(block.id)
                    pending_tool_uses.append(block)

            result_msgs.append(msg)
            result_ids.append(msg_id)

        # User messages with tool_results: validate and mark as satisfied
        elif msg.role == Role.USER and isinstance(msg.content, list):
            has_tool_results = any(isinstance(b, ToolResult) for b in msg.content)

            if not has_tool_results:
                result_msgs.append(msg)
                result_ids.append(msg_id)
                continue

            # Filter to only tool_results with matching tool_uses
            valid_content: list[ContentBlock] = []
            for block in msg.content:
                if not isinstance(block, ToolResult):
                    valid_content.append(block)
                elif block.tool_use_id in seen_tool_use_ids:
                    valid_content.append(block)
                    pending_tool_uses[:] = [
                        tu for tu in pending_tool_uses if tu.id != block.tool_use_id
                    ]
                else:
                    logger.warning(
                        "Removing orphaned tool_result: %s", block.tool_use_id
                    )

            if valid_content:
                result_msgs.append(Message(role=msg.role, content=valid_content))
                result_ids.append(msg_id)
        else:
            result_msgs.append(msg)
            result_ids.append(msg_id)

    flush_pending_tool_uses()
    return result_msgs, result_ids
