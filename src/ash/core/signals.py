"""Protocol signal constants for agent communication."""

import re

NO_REPLY = "[NO_REPLY]"

# Matches [NO_REPLY] or NO_REPLY (case-insensitive, optional brackets)
_NO_REPLY_PATTERN = re.compile(r"^\[?NO_REPLY\]?$", re.IGNORECASE)


def is_no_reply(text: str) -> bool:
    """Check if text is a NO_REPLY signal.

    Matches the first non-blank line against [NO_REPLY] or NO_REPLY
    (case-insensitive, optional brackets). Any subsequent lines are
    treated as leaked reasoning and still count as NO_REPLY.
    """
    first_line = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break

    if not first_line:
        return False

    return bool(_NO_REPLY_PATTERN.match(first_line))


def contains_no_reply(text: str) -> bool:
    """Check if text contains a NO_REPLY signal anywhere.

    Used for streaming suppression — detects the token even when
    additional content has been appended after it.
    """
    return bool(re.search(r"\[?NO_REPLY\]?", text, re.IGNORECASE))
