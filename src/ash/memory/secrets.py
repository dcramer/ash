"""Canonical secret detection patterns for the memory subsystem.

Used by both GraphStore (add_memory rejection) and MemoryExtractor
(extraction filtering) to prevent storing credentials and secrets.
"""

import re

# Regex patterns for detecting secrets/credentials in memory content
SECRETS_PATTERNS = [
    # API keys and tokens
    re.compile(r"\b(sk-[a-zA-Z0-9]{20,})\b"),  # OpenAI/Anthropic keys
    re.compile(r"\b(gh[pors]_[a-zA-Z0-9]{36,})\b"),  # GitHub tokens (PAT, OAuth, etc.)
    re.compile(r"\b(AKIA[A-Z0-9]{16})\b"),  # AWS access key IDs
    re.compile(r"\b(xox[baprs]-[a-zA-Z0-9-]{10,})\b"),  # Slack tokens
    # Credit card numbers (16 digits with optional separators)
    re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    # Social Security Numbers
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Passwords in common formats
    re.compile(r"\b(?:password|passwd|pwd)\s*(?:is|:|=)\s*\S+", re.IGNORECASE),
    # Private keys
    re.compile(
        r"-----BEGIN\s+(?:RSA\s+|DSA\s+|EC\s+|OPENSSH\s+|PGP\s+)?PRIVATE\s+KEY-----",
        re.IGNORECASE,
    ),
    # API key assignments
    re.compile(
        r"\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*(?:is|:|=)\s*\S+",
        re.IGNORECASE,
    ),
]


def contains_secret(content: str) -> bool:
    """Check if content contains patterns that look like secrets.

    Args:
        content: The text content to check.

    Returns:
        True if any secret pattern is detected.
    """
    return any(pattern.search(content) for pattern in SECRETS_PATTERNS)
