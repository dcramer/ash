"""Schedule handler for processing scheduled tasks."""

import logging
from collections.abc import Awaitable, Callable
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
        scheduled_by = f"@{entry.username}" if entry.username else "unknown"
        prefixed_message = (
            f"[SCHEDULED TASK - scheduled by {scheduled_by}]\n\n"
            f"Message to deliver: {entry.message}\n\n"
            f"Instructions: Send this message to the chat. If the message mentions a specific "
            f"person (like @someone), that's who it's for. If it's a reminder, just deliver it. "
            f"If it asks you to do something (like check weather, run a command), do it and "
            f"report the result. Keep your response concise."
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
