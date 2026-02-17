"""Schedule handler for processing scheduled tasks."""

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo

from ash.agents.types import ChildActivated
from ash.core.session import SessionState
from ash.core.signals import is_no_reply
from ash.scheduling.types import ScheduleEntry

if TYPE_CHECKING:
    from ash.agents.executor import AgentExecutor
    from ash.agents.types import StackFrame
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
- Respond with exactly: [NO_REPLY]
- Do NOT send a message explaining the skip

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


class MessageSender(Protocol):
    """Protocol for sending messages to a chat. Returns the sent message ID."""

    async def __call__(
        self, chat_id: str, text: str, *, reply_to: str | None = None
    ) -> str: ...


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
        agent_executor: "AgentExecutor | None" = None,
    ):
        """Initialize the handler.

        Args:
            agent: Agent instance to process tasks.
            senders: Map of provider name -> send function.
            registrars: Map of provider name -> message registrar for thread tracking.
            timezone: Fallback IANA timezone for computing fire times.
            agent_executor: Optional executor for running subagent skill loops.
        """
        self._agent = agent
        self._senders = senders
        self._registrars = registrars or {}
        self._timezone = timezone
        self._agent_executor = agent_executor

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
                "scheduled_task_missing_routing",
                extra={
                    "schedule.entry_id": entry.id,
                    "messaging.provider": entry.provider,
                    "messaging.chat_id": entry.chat_id,
                },
            )
            raise ValueError("Missing required routing context (provider/chat_id)")

        logger.info(
            "scheduled_task_executing",
            extra={
                "schedule.entry_id": entry.id,
                "schedule.message_preview": entry.message[:50],
                "messaging.provider": entry.provider,
                "messaging.chat_id": entry.chat_id,
                "messaging.chat_title": entry.chat_title,
            },
        )

        # Compute fire time and delay
        fire_time = entry.previous_fire_time(self._timezone) or datetime.now(UTC)
        delay_seconds = (datetime.now(UTC) - fire_time).total_seconds()

        # Format times in entry's timezone
        tz = ZoneInfo(entry.timezone or self._timezone or "UTC")
        current_time_str = datetime.now(tz).strftime("%Y-%m-%d %H:%M")
        fire_time_str = fire_time.astimezone(tz).strftime("%Y-%m-%d %H:%M")

        logger.info(
            "scheduled_task_timing",
            extra={
                "schedule.entry_id": entry.id,
                "schedule.fire_time": fire_time_str,
                "schedule.current_time": current_time_str,
                "schedule.delay": format_delay(delay_seconds),
            },
        )

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
        # Populate context so system prompt builder includes full context
        session.context.username = entry.username or ""
        session.context.is_scheduled_task = True
        if entry.chat_title:
            session.context.chat_title = entry.chat_title

        try:
            # Process through agent
            # Note: ToolContext is created internally by agent using session fields
            response = await self._agent.process_message(
                prefixed_message,
                session,
                user_id=entry.user_id,
            )

            if response.text and not is_no_reply(response.text):
                await self._send_response(entry, session, response.text)
            else:
                logger.info("scheduled_task_no_response")

        except ChildActivated as ca:
            # A skill was invoked — run the subagent loop to completion
            if self._agent_executor and ca.main_frame and ca.child_frame:
                result_text = await self._run_skill_loop(
                    ca.main_frame, ca.child_frame, session
                )
                if result_text and not is_no_reply(result_text):
                    await self._send_response(entry, session, result_text)
            else:
                logger.error("scheduled_task_no_executor")

        except Exception as e:
            logger.error(
                "scheduled_task_failed", extra={"error.message": str(e)}, exc_info=True
            )

    async def _send_response(
        self,
        entry: ScheduleEntry,
        session: SessionState,
        text: str,
    ) -> None:
        """Send a response message back to the chat that scheduled the task."""
        if not entry.chat_id or not entry.provider:
            return

        sender = self._senders.get(entry.provider)
        if not sender:
            logger.warning("no_sender_configured", extra={"provider": entry.provider})
            return

        response_text = text
        if entry.username:
            response_text = f"@{entry.username} {response_text}"

        reply_to = session.context.reply_to_message_id
        message_id = await sender(entry.chat_id, response_text, reply_to=reply_to)
        logger.info(
            "scheduled_response_sent",
            extra={
                "messaging.provider": entry.provider,
                "messaging.chat_id": entry.chat_id,
                "response.preview": response_text[:50],
            },
        )

        registrar = self._registrars.get(entry.provider)
        if registrar:
            await registrar(entry.chat_id, message_id)
            logger.debug(f"Registered scheduled message {message_id} in thread index")

    async def _run_skill_loop(
        self,
        main_frame: "StackFrame",
        child_frame: "StackFrame",
        session: SessionState,
    ) -> str | None:
        """Run a subagent skill loop to completion without user interaction.

        Pushes the main and child frames onto a temporary stack and drives
        execute_turn until the stack unwinds back to the main agent producing
        a final text response.

        Returns:
            Collected text output, or None if no output was produced.
        """
        from ash.agents.types import AgentStack, TurnAction

        assert self._agent_executor is not None

        stack = AgentStack()
        stack.push(main_frame)
        stack.push(child_frame)

        collected_text: list[str] = []
        max_turns = 50  # Safety limit

        for _ in range(max_turns):
            top = stack.top
            if top is None:
                break

            is_main = stack.depth == 1

            # First turn for a newly pushed child has no user_message/tool_result
            result = await self._agent_executor.execute_turn(top)

            if result.action == TurnAction.COMPLETE:
                # Subagent finished — pop it and feed result to parent
                completed = stack.pop()
                parent = stack.top
                if parent and completed.parent_tool_use_id:
                    # Feed result back as tool_result for the parent's tool_use
                    result2 = await self._agent_executor.execute_turn(
                        parent,
                        tool_result=(
                            completed.parent_tool_use_id,
                            result.text,
                            False,
                        ),
                    )
                    # Process the parent's response
                    if result2.action == TurnAction.SEND_TEXT:
                        if stack.depth == 1:
                            collected_text.append(result2.text)
                            break
                    elif result2.action == TurnAction.COMPLETE:
                        stack.pop()
                        if result2.text:
                            collected_text.append(result2.text)
                        break
                    elif result2.action == TurnAction.CHILD_ACTIVATED:
                        if result2.child_frame:
                            stack.push(result2.child_frame)
                    elif result2.action in (
                        TurnAction.ERROR,
                        TurnAction.MAX_ITERATIONS,
                    ):
                        logger.error(
                            "parent_agent_error_after_skill",
                            extra={"error.message": result2.text},
                        )
                        stack.pop()
                        break
                    # INTERRUPT in scheduled mode — no user to interact with
                    elif result2.action == TurnAction.INTERRUPT:
                        logger.warning("agent_interrupted_scheduled_mode")
                        break
                else:
                    # No parent or no tool_use_id — just collect text
                    if result.text:
                        collected_text.append(result.text)
                    break

            elif result.action == TurnAction.SEND_TEXT:
                if is_main:
                    # Main agent produced text — final output
                    collected_text.append(result.text)
                    break
                # Subagent "sent text" — in scheduled mode, treat as completion
                # since there's no user to interact with
                completed = stack.pop()
                parent = stack.top
                if parent and completed.parent_tool_use_id:
                    result2 = await self._agent_executor.execute_turn(
                        parent,
                        tool_result=(
                            completed.parent_tool_use_id,
                            result.text,
                            False,
                        ),
                    )
                    if result2.action == TurnAction.SEND_TEXT:
                        if stack.depth == 1:
                            collected_text.append(result2.text)
                            break
                    elif result2.action == TurnAction.COMPLETE:
                        stack.pop()
                        if result2.text:
                            collected_text.append(result2.text)
                        break
                    elif result2.action == TurnAction.CHILD_ACTIVATED:
                        if result2.child_frame:
                            stack.push(result2.child_frame)
                    elif result2.action in (
                        TurnAction.ERROR,
                        TurnAction.MAX_ITERATIONS,
                    ):
                        logger.error(
                            "parent_agent_error_after_subagent",
                            extra={"error.message": result2.text},
                        )
                        stack.pop()
                        break
                else:
                    if result.text:
                        collected_text.append(result.text)
                    break

            elif result.action == TurnAction.CHILD_ACTIVATED:
                if result.child_frame:
                    stack.push(result.child_frame)

            elif result.action == TurnAction.INTERRUPT:
                # No user in scheduled mode — feed error to parent
                logger.warning("subagent_interrupted_scheduled_mode")
                completed = stack.pop()
                parent = stack.top
                if parent and completed.parent_tool_use_id:
                    await self._agent_executor.execute_turn(
                        parent,
                        tool_result=(
                            completed.parent_tool_use_id,
                            "Error: agent requires user interaction but running in scheduled mode",
                            True,
                        ),
                    )
                break

            elif result.action in (TurnAction.ERROR, TurnAction.MAX_ITERATIONS):
                logger.error(
                    "subagent_error_scheduled_mode",
                    extra={
                        "agent.action": result.action.name,
                        "error.message": result.text,
                    },
                )
                completed = stack.pop()
                parent = stack.top
                if parent and completed.parent_tool_use_id:
                    await self._agent_executor.execute_turn(
                        parent,
                        tool_result=(
                            completed.parent_tool_use_id,
                            f"Skill error: {result.text}",
                            True,
                        ),
                    )
                break

        return "\n\n".join(collected_text) if collected_text else None
