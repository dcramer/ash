"""Chat-level history recording.

Tracks ALL messages (user + bot) across all threads in a chat.
Replaces incoming.py — provides a unified history.jsonl per chat.

Format: specs/sessions.md#historyjsonl
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel

from ash.config.paths import get_chat_dir

logger = logging.getLogger(__name__)


class HistoryEntry(BaseModel):
    """Validated schema for history.jsonl entries.

    Spec: specs/sessions.md#historyjsonl
    """

    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    user_id: str | None = None
    username: str | None = None
    display_name: str | None = None
    metadata: dict[str, Any] | None = None


class ChatHistoryWriter:
    """Appends history entries to a chat-level history.jsonl file."""

    def __init__(self, provider: str, chat_id: str):
        self._chat_dir = get_chat_dir(provider, chat_id)
        self._file = self._chat_dir / "history.jsonl"

    def record_user_message(
        self,
        *,
        content: str,
        created_at: datetime | None = None,
        user_id: str | None = None,
        username: str | None = None,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a user message to chat history."""
        entry = HistoryEntry(
            id=str(uuid.uuid4()),
            role="user",
            content=content,
            created_at=created_at or datetime.now(UTC),
            user_id=user_id,
            username=username,
            display_name=display_name,
            metadata=metadata,
        )
        self._append(entry)

    def record_bot_message(
        self,
        *,
        content: str,
        created_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a bot response to chat history."""
        entry = HistoryEntry(
            id=str(uuid.uuid4()),
            role="assistant",
            content=content,
            created_at=created_at or datetime.now(UTC),
            metadata=metadata,
        )
        self._append(entry)

    def _append(self, entry: HistoryEntry) -> None:
        """Append a validated entry to the JSONL file."""
        self._file.parent.mkdir(parents=True, exist_ok=True)
        line = entry.model_dump_json() + "\n"
        with self._file.open("a") as f:
            f.write(line)


def read_recent_chat_history(
    provider: str,
    chat_id: str,
    limit: int = 15,
) -> list[HistoryEntry]:
    """Read recent entries from a chat-level history.jsonl.

    Args:
        provider: Provider name (e.g., "telegram").
        chat_id: Chat identifier.
        limit: Maximum number of entries to return.

    Returns:
        List of validated HistoryEntry instances (most recent last).
    """
    chat_dir = get_chat_dir(provider, chat_id)
    history_file = chat_dir / "history.jsonl"

    if not history_file.exists():
        return []

    try:
        lines = history_file.read_text().strip().split("\n")
        entries: list[HistoryEntry] = []
        for line in lines[-limit:]:
            try:
                entries.append(HistoryEntry.model_validate_json(line))
            except Exception:
                logger.debug("Skipping invalid history line: %s", line[:80])
                continue
        return entries
    except Exception as e:
        logger.debug("Failed to read chat history: %s", e)
        return []
