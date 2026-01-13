"""Schedule handler for processing scheduled tasks."""

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from ash.core.session import SessionState
from ash.events.schedule import ScheduleEntry

if TYPE_CHECKING:
    from ash.core.agent import Agent

logger = logging.getLogger(__name__)

# Type for message sender: (chat_id, text) -> message_id
MessageSender = Callable[[str, str], Awaitable[str]]


class ScheduledTaskHandler:
    """Handles execution of scheduled tasks.

    Processes scheduled entries by:
    1. Creating an ephemeral session for the task
    2. Running the message through the agent
    3. Sending the response back via the appropriate provider
    """

    def __init__(
        self,
        agent: "Agent",
        senders: dict[str, MessageSender],
    ):
        """Initialize the handler.

        Args:
            agent: Agent instance to process tasks.
            senders: Map of provider name -> send function.
        """
        self._agent = agent
        self._senders = senders

    async def handle(self, entry: ScheduleEntry) -> None:
        """Process a scheduled task.

        Args:
            entry: The schedule entry to process.

        Raises:
            ValueError: If required routing context is missing.
        """
        # Validate routing context
        if not entry.provider or not entry.chat_id:
            logger.error(
                f"Skipping scheduled task - missing routing context: "
                f"provider={entry.provider!r}, chat_id={entry.chat_id!r}"
            )
            raise ValueError("Missing required routing context (provider/chat_id)")

        logger.info(
            f"Executing scheduled task: {entry.message[:50]}... "
            f"(provider={entry.provider}, chat_id={entry.chat_id})"
        )

        # Build context for the agent
        # The critical requirement is that the agent MUST produce text output that gets
        # sent to the user. The user scheduled this and expects a notification.
        created_at = entry.created_at.isoformat() if entry.created_at else "unknown"
        user_ref = f"@{entry.username}" if entry.username else "the user"
        now = datetime.now(UTC).isoformat()
        prefixed_message = (
            f"[SCHEDULED NOTIFICATION for {user_ref}]\n"
            f"Originally scheduled at {created_at}, now triggered at {now}.\n\n"
            f"Task: {entry.message}\n\n"
            f"CRITICAL: {user_ref} expects to be notified. You MUST end with a text message "
            f"that will be sent to them. If this is a reminder, deliver it. If this is an action, "
            f"do it and report the result. Your final response goes directly to {user_ref}."
        )

        # Create ephemeral session for this task
        session = SessionState(
            session_id=f"scheduled_{uuid4().hex[:8]}",
            provider=entry.provider or "scheduled",
            chat_id=entry.chat_id or "",
            user_id=entry.user_id or "",
        )

        try:
            # Process through agent
            # Note: ToolContext is created internally by agent using session fields
            response = await self._agent.process_message(
                prefixed_message,
                session,
                user_id=entry.user_id,
            )

            # Send response back
            if response.text and entry.chat_id and entry.provider:
                sender = self._senders.get(entry.provider)
                if sender:
                    # Prepend @mention if we have a username
                    response_text = response.text
                    if entry.username:
                        response_text = f"@{entry.username} {response_text}"

                    await sender(entry.chat_id, response_text)
                    logger.info(
                        f"Sent scheduled response to {entry.provider}/{entry.chat_id}: "
                        f"{response_text[:50]}..."
                    )
                else:
                    logger.warning(
                        f"No sender configured for provider: {entry.provider}"
                    )
            elif not response.text:
                logger.info("Scheduled task completed with no response to send")

        except Exception as e:
            logger.error(f"Scheduled task failed: {e}", exc_info=True)
