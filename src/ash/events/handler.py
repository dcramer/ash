"""Schedule handler for processing scheduled tasks."""

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4
from zoneinfo import ZoneInfo

from ash.core.session import SessionState
from ash.events.schedule import ScheduleEntry

if TYPE_CHECKING:
    from ash.core.agent import Agent

logger = logging.getLogger(__name__)

SCHEDULED_TASK_WRAPPER = """\
You are executing a scheduled task. Before running the task, evaluate whether it's still relevant given the delay.

<context>
Entry ID: {entry_id}
{schedule_line}
Scheduled by: {scheduled_by}
</context>

<timing>
Current time: {current_time}
Scheduled fire time: {fire_time}
Delay: {delay_human}
</timing>

<decision-guidance>
## Step 1: Classify the task

TIME-SENSITIVE tasks depend on being run close to their scheduled time:
- Greetings tied to time of day ("good morning", "good night")
- Reminders for specific moments ("remind me at 2pm to call")
- Event prompts ("daily standup", "weekly sync reminder")

TIME-INDEPENDENT tasks provide value regardless of when they run:
- Data fetching (weather, transit, stocks, news)
- Reports and summaries
- Backups and syncs
- General reminders without time context

## Step 2: Decide whether to execute

For TIME-SENSITIVE tasks:
- If delay > 2 hours AND the task's meaning has passed: SKIP
- If delay > 4 hours: Almost certainly SKIP unless task is clearly still useful
- Use judgment for delays between 30 min - 2 hours

For TIME-INDEPENDENT tasks:
- Always EXECUTE regardless of delay

## Step 3: What to output

If EXECUTING:
- Run the task normally
- Do NOT mention the delay unless it affects the task content

If SKIPPING:
- Briefly explain why (one sentence)
- If recurring, mention it will run at the next scheduled time
- Example: "Skipping morning greeting - it's now 3:45 PM. This runs daily at 8 AM."

Do NOT apologize for delays or explain scheduling mechanics.
</decision-guidance>

<task>
{message}
</task>"""


def format_delay(seconds: float) -> str:
    """Format delay in human-readable form."""
    minutes = seconds / 60
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"~{int(minutes)} minutes"
    hours = minutes / 60
    if hours < 24:
        return f"~{hours:.1f} hours"
    days = hours / 24
    return f"~{days:.1f} days"


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
        timezone: str = "UTC",
    ):
        """Initialize the handler.

        Args:
            agent: Agent instance to process tasks.
            senders: Map of provider name -> send function.
            registrars: Map of provider name -> message registrar for thread tracking.
            timezone: Fallback IANA timezone for computing fire times.
        """
        self._agent = agent
        self._senders = senders
        self._registrars = registrars or {}
        self._timezone = timezone

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

        chat_display = (
            f"{entry.chat_title} ({entry.chat_id})"
            if entry.chat_title
            else entry.chat_id
        )
        logger.info(
            f"Executing scheduled task: {entry.message[:50]}... "
            f"(provider={entry.provider}, chat={chat_display})"
        )

        # Compute fire time and delay
        fire_time = entry.previous_fire_time(self._timezone) or datetime.now(UTC)
        delay_seconds = (datetime.now(UTC) - fire_time).total_seconds()

        # Format times in entry's timezone
        tz = ZoneInfo(entry.timezone or self._timezone or "UTC")
        current_time_str = datetime.now(tz).strftime("%Y-%m-%d %H:%M")
        fire_time_str = fire_time.astimezone(tz).strftime("%Y-%m-%d %H:%M")

        # Build schedule line
        if entry.cron:
            schedule_line = f"Schedule: {entry.cron} (recurring)"
        else:
            schedule_line = f"Trigger: {fire_time_str} (one-shot)"

        # Build wrapped message with timing context
        prefixed_message = SCHEDULED_TASK_WRAPPER.format(
            entry_id=entry.id,
            schedule_line=schedule_line,
            scheduled_by=f"@{entry.username}" if entry.username else "unknown",
            current_time=current_time_str,
            fire_time=fire_time_str,
            delay_human=format_delay(delay_seconds),
            message=entry.message,
        )

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
        if entry.chat_title:
            session.metadata["chat_title"] = entry.chat_title

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
