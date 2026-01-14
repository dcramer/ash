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

DEFAULT_RECENCY_WINDOW = 10


def fit_messages_to_budget(
    messages: list[Message],
    token_counts: list[int],
    budget: int,
    message_ids: list[str] | None = None,
) -> tuple[list[Message], list[str]]:
    if message_ids is None:
        message_ids = [""] * len(messages)

    result_msgs: list[Message] = []
    result_ids: list[str] = []
    remaining = budget

    for msg, msg_id, tokens in zip(
        reversed(messages), reversed(message_ids), reversed(token_counts), strict=False
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
    if message_ids is None:
        message_ids = [""] * len(messages)

    if token_budget is None or not messages:
        return messages.copy(), list(message_ids)

    n = len(messages)
    recency_start = max(0, n - recency_window)
    recency_tokens = sum(token_counts[recency_start:])

    if recency_tokens >= token_budget:
        pruned_msgs, pruned_ids = fit_messages_to_budget(
            messages[recency_start:],
            token_counts[recency_start:],
            token_budget,
            message_ids[recency_start:],
        )
        return validate_tool_pairs(pruned_msgs, pruned_ids)

    remaining_budget = token_budget - recency_tokens
    included_msgs: list[Message] = []
    included_ids: list[str] = []

    for msg, msg_id, tokens in zip(
        reversed(messages[:recency_start]),
        reversed(message_ids[:recency_start]),
        reversed(token_counts[:recency_start]),
        strict=False,
    ):
        if tokens <= remaining_budget:
            included_msgs.insert(0, msg)
            included_ids.insert(0, msg_id)
            remaining_budget -= tokens
        else:
            break

    combined_msgs = included_msgs + messages[recency_start:]
    combined_ids = included_ids + message_ids[recency_start:]
    return validate_tool_pairs(combined_msgs, combined_ids)


def content_block_to_dict(block: ContentBlock) -> dict[str, Any]:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUse):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResult):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    return {}


def content_block_from_dict(data: dict[str, Any]) -> ContentBlock | None:
    block_type = data.get("type")
    if block_type == "text":
        return TextContent(text=data["text"])
    if block_type == "tool_use":
        return ToolUse(id=data["id"], name=data["name"], input=data["input"])
    if block_type == "tool_result":
        return ToolResult(
            tool_use_id=data["tool_use_id"],
            content=data["content"],
            is_error=data.get("is_error", False),
        )
    return None


def _make_synthetic_results(tool_uses: list[ToolUse]) -> Message:
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
    if not messages:
        return messages, message_ids or []

    if message_ids is None:
        message_ids = [""] * len(messages)

    result_msgs: list[Message] = []
    result_ids: list[str] = []
    pending_tool_uses: list[ToolUse] = []
    seen_tool_use_ids: set[str] = set()

    def flush_pending() -> None:
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
        if msg.role == Role.ASSISTANT and isinstance(msg.content, list):
            flush_pending()
            for block in msg.content:
                if isinstance(block, ToolUse):
                    seen_tool_use_ids.add(block.id)
                    pending_tool_uses.append(block)
            result_msgs.append(msg)
            result_ids.append(msg_id)

        elif msg.role == Role.USER and isinstance(msg.content, list):
            if not any(isinstance(b, ToolResult) for b in msg.content):
                result_msgs.append(msg)
                result_ids.append(msg_id)
                continue

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
                    logger.debug(
                        "Filtering orphaned tool_result: %s", block.tool_use_id
                    )

            if valid_content:
                result_msgs.append(Message(role=msg.role, content=valid_content))
                result_ids.append(msg_id)

        elif msg.content:
            result_msgs.append(msg)
            result_ids.append(msg_id)

    flush_pending()
    return result_msgs, result_ids
