"""Schedule task tool for creating scheduled tasks with context."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ash.tools.base import Tool, ToolContext, ToolResult


class ScheduleTaskTool(Tool):
    """Schedule a task for future execution.

    Writes to the schedule file with full context (chat_id, user_id, provider)
    so responses can be routed back to the originating conversation.
    """

    def __init__(self, schedule_file: Path) -> None:
        """Initialize schedule tool.

        Args:
            schedule_file: Path to schedule.jsonl file.
        """
        self._schedule_file = schedule_file

    @property
    def name(self) -> str:
        return "schedule_task"

    @property
    def description(self) -> str:
        return (
            "Schedule a task for future execution. Use trigger_at for one-time tasks, "
            "or cron for recurring tasks. The response will be sent back to this conversation."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": (
                        "The complete prompt to process when triggered. Must be self-contained "
                        "and make sense without additional context. "
                        "For reminders: 'Remind the user to [action]' (e.g., 'Remind the user to wake up their wife'). "
                        "For tasks: describe what to do (e.g., 'Check the build status and report results')."
                    ),
                },
                "trigger_at": {
                    "type": "string",
                    "description": (
                        "ISO 8601 timestamp (UTC) for one-time execution. "
                        "Example: 2026-01-12T09:00:00Z"
                    ),
                },
                "cron": {
                    "type": "string",
                    "description": (
                        "Cron expression for recurring execution (5-field format). "
                        "Examples: '0 8 * * *' (daily 8am), '0 9 * * 1' (Mondays 9am)"
                    ),
                },
            },
            "required": ["message"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Create a scheduled task.

        Args:
            input_data: Must contain 'message' and either 'trigger_at' or 'cron'.
            context: Execution context with chat/user info.

        Returns:
            Tool result with confirmation or error.
        """
        message = input_data.get("message")
        trigger_at = input_data.get("trigger_at")
        cron = input_data.get("cron")

        if not message:
            return ToolResult.error("Missing required parameter: message")

        # Scheduling requires a provider with persistent chat to route response back
        if not context.provider or not context.chat_id:
            return ToolResult.error(
                "Scheduling requires a provider with persistent chat (e.g., Telegram). "
                "Cannot schedule tasks from CLI."
            )

        if not trigger_at and not cron:
            return ToolResult.error(
                "Must specify either 'trigger_at' (one-time) or 'cron' (recurring)"
            )

        if trigger_at and cron:
            return ToolResult.error(
                "Cannot specify both 'trigger_at' and 'cron'. Choose one."
            )

        # Validate trigger_at format
        if trigger_at:
            try:
                parsed_time = datetime.fromisoformat(trigger_at.replace("Z", "+00:00"))
                if parsed_time <= datetime.now(UTC):
                    return ToolResult.error(
                        f"trigger_at must be in the future. Got: {trigger_at}"
                    )
            except ValueError as e:
                return ToolResult.error(f"Invalid trigger_at format: {e}")

        # Validate cron format
        if cron:
            try:
                from croniter import croniter

                croniter(cron)  # Validates the expression
            except Exception as e:
                return ToolResult.error(f"Invalid cron expression: {e}")

        # Build entry with context
        entry: dict[str, Any] = {"message": message}

        if trigger_at:
            entry["trigger_at"] = trigger_at
        if cron:
            entry["cron"] = cron

        # Inject context for routing response
        if context.chat_id:
            entry["chat_id"] = context.chat_id
        if context.user_id:
            entry["user_id"] = context.user_id
        if context.metadata.get("username"):
            entry["username"] = context.metadata["username"]
        if context.provider:
            entry["provider"] = context.provider

        entry["created_at"] = datetime.now(UTC).isoformat()

        # Ensure parent directory exists
        self._schedule_file.parent.mkdir(parents=True, exist_ok=True)

        # Append to schedule file
        try:
            with self._schedule_file.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            return ToolResult.error(f"Failed to write schedule: {e}")

        # Format confirmation
        msg_preview = f"{message[:50]}..." if len(message) > 50 else message
        if trigger_at:
            return ToolResult.success(
                f"Scheduled one-time task for {trigger_at}: {msg_preview}"
            )
        return ToolResult.success(f"Scheduled recurring task ({cron}): {msg_preview}")
