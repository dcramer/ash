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

# Type for message registrar: (chat_id, message_id) -> None
# Registers a sent message in the thread index so replies are tracked
MessageRegistrar = Callable[[str, str], Awaitable[None]]


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
        registrars: dict[str, MessageRegistrar] | None = None,
    ):
        """Initialize the handler.

        Args:
            agent: Agent instance to process tasks.
            senders: Map of provider name -> send function.
            registrars: Map of provider name -> message registrar for thread tracking.
        """
        self._agent = agent
        self._senders = senders
        self._registrars = registrars or {}

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

        # Build schedule context as facts (system prompt handles instructions)
        if entry.cron:
            schedule_line = f"Schedule: {entry.cron} (recurring)"
        else:
            trigger_str = (
                entry.trigger_at.isoformat() if entry.trigger_at else "unknown"
            )
            schedule_line = f"Trigger: {trigger_str} (one-shot)"

        scheduled_by = f"@{entry.username}" if entry.username else "unknown"

        # Present as context facts, similar to system prompt sections
        schedule_context = (
            f"[Scheduled Task]\n"
            f"Entry ID: {entry.id}\n"
            f"{schedule_line}\n"
            f"Scheduled by: {scheduled_by}\n"
        )

        # User message is just the task - system prompt handles behavior
        prefixed_message = f"{schedule_context}\n{entry.message}"

        # Create ephemeral session for this task
        session = SessionState(
            session_id=f"scheduled_{uuid4().hex[:8]}",
            provider=entry.provider or "scheduled",
            chat_id=entry.chat_id or "",
            user_id=entry.user_id or "",
        )
        # Populate metadata so system prompt builder includes full context
        session.metadata["username"] = entry.username or ""
        session.metadata["session_mode"] = "fresh"
        session.metadata["is_scheduled_task"] = True

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

                    message_id = await sender(entry.chat_id, response_text)
                    logger.info(
                        f"Sent scheduled response to {entry.provider}/{entry.chat_id}: "
                        f"{response_text[:50]}..."
                    )

                    # Register the message in thread index so replies get tracked
                    registrar = self._registrars.get(entry.provider)
                    if registrar:
                        await registrar(entry.chat_id, message_id)
                        logger.debug(
                            f"Registered scheduled message {message_id} in thread index"
                        )
                else:
                    logger.warning(
                        f"No sender configured for provider: {entry.provider}"
                    )
            elif not response.text:
                logger.info("Scheduled task completed with no response to send")

        except Exception as e:
            logger.error(f"Scheduled task failed: {e}", exc_info=True)
