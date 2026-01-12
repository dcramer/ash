"""Context compaction for managing conversation length.

Compaction summarizes older messages when context gets too large,
preserving important information while staying within token limits.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ash.core.tokens import estimate_message_tokens
from ash.llm.types import Message, Role, TextContent

if TYPE_CHECKING:
    from ash.llm import LLMProvider

logger = logging.getLogger(__name__)

# Prefix/suffix for compaction summaries (helps LLM understand context)
COMPACTION_PREFIX = "[Previous conversation summary]\n"
COMPACTION_SUFFIX = "\n[End of summary - conversation continues below]"


@dataclass
class CompactionSettings:
    """Settings for when and how to compact."""

    enabled: bool = True
    reserve_tokens: int = 16384  # Buffer to leave free
    keep_recent_tokens: int = 20000  # Always keep recent context
    summary_max_tokens: int = 2000  # Max tokens for summary


@dataclass
class CompactionResult:
    """Result of a compaction operation."""

    summary: str
    tokens_before: int
    tokens_after: int
    messages_removed: int
    first_kept_index: int


def should_compact(
    context_tokens: int,
    context_window: int,
    settings: CompactionSettings,
) -> bool:
    """Check if compaction is needed.

    Args:
        context_tokens: Current context token count.
        context_window: Model's context window size.
        settings: Compaction settings.

    Returns:
        True if compaction should be triggered.
    """
    if not settings.enabled:
        return False
    return context_tokens > context_window - settings.reserve_tokens


def find_compaction_boundary(
    messages: list[Message],
    token_counts: list[int],
    keep_recent_tokens: int,
) -> int:
    """Find the index where compaction should split messages.

    Everything before this index will be summarized.
    Everything from this index onward will be kept.

    Args:
        messages: List of messages.
        token_counts: Token counts per message.
        keep_recent_tokens: Minimum tokens to keep in recent context.

    Returns:
        Index of first message to keep (0 means no compaction possible).
    """
    if not messages or not token_counts:
        return 0

    # Work backward from the end to find how much to keep
    total_kept = 0
    keep_from = len(messages)

    for i in range(len(messages) - 1, -1, -1):
        total_kept += token_counts[i]
        if total_kept >= keep_recent_tokens:
            keep_from = i
            break
        keep_from = i

    # Need at least some messages to summarize
    if keep_from <= 1:
        return 0

    return keep_from


async def generate_summary(
    messages: list[Message],
    llm: "LLMProvider",
    model: str | None = None,
    max_tokens: int = 2000,
) -> str:
    """Generate a summary of messages using the LLM.

    Args:
        messages: Messages to summarize.
        llm: LLM provider for generating summary.
        model: Model to use (None = provider default).
        max_tokens: Maximum tokens for summary.

    Returns:
        Summary text.
    """
    # Build a prompt with the messages to summarize
    conversation_text = []
    for msg in messages:
        role = msg.role.value.upper()
        if isinstance(msg.content, str):
            text = msg.content
        else:
            # Extract text from content blocks
            texts = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    texts.append(block.text)
            text = "\n".join(texts) if texts else "[non-text content]"

        # Truncate very long messages
        if len(text) > 1000:
            text = text[:1000] + "..."

        conversation_text.append(f"{role}: {text}")

    prompt = (
        "Summarize the following conversation concisely. "
        "Focus on key information, decisions made, and important context. "
        "Do not include greetings or filler. Be direct and factual.\n\n"
        + "\n\n".join(conversation_text)
    )

    response = await llm.complete(
        messages=[Message(role=Role.USER, content=prompt)],
        model=model,
        max_tokens=max_tokens,
        temperature=0.3,  # Lower temperature for more consistent summaries
    )

    return response.message.get_text() or ""


def create_summary_message(summary: str) -> Message:
    """Create a message containing the compaction summary.

    Args:
        summary: The generated summary text.

    Returns:
        A user message with the summary.
    """
    content = COMPACTION_PREFIX + summary + COMPACTION_SUFFIX
    return Message(role=Role.USER, content=content)


async def compact_messages(
    messages: list[Message],
    token_counts: list[int],
    llm: "LLMProvider",
    settings: CompactionSettings,
    model: str | None = None,
) -> tuple[list[Message], list[int], CompactionResult | None]:
    """Compact messages by summarizing older ones.

    Args:
        messages: Current messages.
        token_counts: Token counts per message.
        llm: LLM provider for generating summary.
        settings: Compaction settings.
        model: Model to use for summary.

    Returns:
        Tuple of (new_messages, new_token_counts, compaction_result).
        Returns None for result if no compaction was performed.
    """
    # Find where to split
    boundary = find_compaction_boundary(
        messages, token_counts, settings.keep_recent_tokens
    )

    if boundary == 0:
        logger.debug("No messages to compact")
        return messages, token_counts, None

    # Split messages
    to_summarize = messages[:boundary]
    to_keep = messages[boundary:]
    kept_tokens = token_counts[boundary:]

    logger.info(
        f"Compacting {len(to_summarize)} messages into summary, keeping {len(to_keep)}"
    )

    # Generate summary
    summary = await generate_summary(
        to_summarize,
        llm,
        model=model,
        max_tokens=settings.summary_max_tokens,
    )

    # Create summary message
    summary_message = create_summary_message(summary)
    summary_tokens = estimate_message_tokens("user", summary_message.content)

    # Build new message list
    new_messages = [summary_message] + to_keep
    new_token_counts = [summary_tokens] + kept_tokens

    tokens_before = sum(token_counts)
    tokens_after = sum(new_token_counts)

    result = CompactionResult(
        summary=summary,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        messages_removed=len(to_summarize),
        first_kept_index=boundary,
    )

    logger.info(
        f"Compaction complete: {tokens_before} -> {tokens_after} tokens "
        f"({len(to_summarize)} messages summarized)"
    )

    return new_messages, new_token_counts, result
