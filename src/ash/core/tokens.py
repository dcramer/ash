"""Token estimation utilities for message pruning."""

import json
from typing import Any


def estimate_tokens(text: str) -> int:
    """Estimate token count using simple heuristic.

    Uses approximation: ~4 characters per token for English text.
    This avoids external dependencies (tiktoken) while being accurate enough
    for pruning decisions.

    Args:
        text: Text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    # ~4 chars per token is a reasonable approximation for English
    return max(1, len(text) // 4 + 1)


def estimate_message_tokens(role: str, content: str | list[Any]) -> int:
    """Estimate tokens for a full message including structure overhead.

    Args:
        role: Message role (user, assistant).
        content: Message content (string or content blocks).

    Returns:
        Estimated token count.
    """
    # Base overhead for message structure (role, delimiters)
    overhead = 4

    if isinstance(content, str):
        return overhead + estimate_tokens(content)

    # Content blocks
    total = overhead
    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "text":
                total += estimate_tokens(block.get("text", ""))
            elif block_type == "tool_use":
                # tool_use: name + JSON input
                total += estimate_tokens(block.get("name", ""))
                total += estimate_tokens(json.dumps(block.get("input", {})))
            elif block_type == "tool_result":
                total += estimate_tokens(block.get("content", ""))
        else:
            # Handle dataclass types (TextContent, ToolUse, ToolResult)
            if hasattr(block, "text"):
                total += estimate_tokens(block.text)
            elif hasattr(block, "name") and hasattr(block, "input"):
                total += estimate_tokens(block.name)
                total += estimate_tokens(json.dumps(block.input))
            elif hasattr(block, "content"):
                total += estimate_tokens(block.content)

    return total
