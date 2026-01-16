"""Incoming message recording for observability.

This module records ALL incoming Telegram messages, regardless of whether they
triggered a bot response. This provides:

1. Full audit trail of group chat activity
2. Debugging info when messages are unexpectedly skipped
3. Metadata about why each message was or wasn't processed

Messages are stored in per-chat JSONL files at:
    ~/.ash/chats/{provider}_{chat_id}/incoming.jsonl

Each record includes:
- was_processed: Whether the message triggered a bot session
- skip_reason: Why the message was skipped (if applicable)
  - "not_mentioned_or_reply": Group message didn't mention bot or reply to conversation
  - "group_not_allowed": Chat ID not in allowed_groups config
  - "user_not_allowed": User not authorized for DMs
  - "no_user": Message had no from_user (system message)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from ash.config.paths import get_chat_dir


@dataclass
class IncomingMessageRecord:
    """Record of an incoming message before processing decision."""

    external_id: str
    chat_id: str
    user_id: str | None
    username: str | None
    display_name: str | None
    text: str | None
    timestamp: str
    was_processed: bool
    skip_reason: str | None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class IncomingMessageWriter:
    """Appends incoming message records to a JSONL file."""

    def __init__(self, provider: str, chat_id: str):
        self._chat_dir = get_chat_dir(provider, chat_id)
        self._file = self._chat_dir / "incoming.jsonl"

    def record(self, record: IncomingMessageRecord) -> None:
        """Append a message record to the JSONL file."""
        self._file.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict()) + "\n"
        with self._file.open("a") as f:
            f.write(line)
