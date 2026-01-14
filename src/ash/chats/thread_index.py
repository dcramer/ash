"""Thread index for reply-chain tracking in group chats.

Maps message external_ids to thread_ids for quick lookup when determining
which session a reply belongs to.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ash.chats.manager import ChatStateManager

logger = logging.getLogger(__name__)


class ThreadIndex:
    """Index mapping message external_ids to thread_ids for a chat.

    In group chats without forum topics, we use reply chains to define threads.
    Each non-reply message starts a new thread, and all replies in that chain
    share the same thread_id (the external_id of the root message).

    The index is backed by ChatState.thread_index for persistence.
    """

    def __init__(self, chat_state_manager: ChatStateManager) -> None:
        self._manager = chat_state_manager
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> dict[str, str]:
        """Ensure state is loaded and return the thread_index dict."""
        state = self._manager.load()
        return state.thread_index

    def resolve_thread_id(
        self,
        external_id: str,
        reply_to_external_id: str | None,
    ) -> str:
        """Determine thread_id for a message.

        Args:
            external_id: The message's external ID (Telegram message_id)
            reply_to_external_id: The external ID of the message being replied to

        Returns:
            The thread_id for this message. If replying to a known message,
            returns the same thread_id. Otherwise returns external_id as a new thread.
        """
        index = self._ensure_loaded()

        if reply_to_external_id:
            parent_thread = index.get(reply_to_external_id)
            if parent_thread:
                logger.debug(
                    "Message %s joins thread %s (via reply to %s)",
                    external_id,
                    parent_thread,
                    reply_to_external_id,
                )
                return parent_thread
            logger.debug(
                "Message %s replied to unknown message %s, starting new thread",
                external_id,
                reply_to_external_id,
            )

        # Start new thread using this message's ID as the thread_id
        logger.debug("Message %s starts new thread", external_id)
        return external_id

    def register_message(self, external_id: str, thread_id: str) -> None:
        """Register a message in a thread.

        Args:
            external_id: The message's external ID
            thread_id: The thread this message belongs to
        """
        with self._lock:
            index = self._ensure_loaded()
            if external_id not in index:
                index[external_id] = thread_id
                self._manager.save()
                logger.debug(
                    "Registered message %s in thread %s", external_id, thread_id
                )

    def get_thread_id(self, external_id: str) -> str | None:
        """Get the thread_id for a message, if known.

        Args:
            external_id: The message's external ID

        Returns:
            The thread_id if the message is registered, None otherwise.
        """
        index = self._ensure_loaded()
        return index.get(external_id)
