"""Protocol signal constants for agent communication."""

NO_REPLY = "[NO_REPLY]"


def is_no_reply(text: str) -> bool:
    """Check if text is a NO_REPLY signal (exact match, whitespace-tolerant)."""
    return text.strip() == NO_REPLY
