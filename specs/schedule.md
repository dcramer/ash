# Schedule System

File-based task scheduling following the "filesystem first" principle.

## Status: Implemented

## Overview

The schedule system allows the agent to schedule future tasks using sandbox CLI commands (`ash schedule create`). The commands write entries to a JSONL file with routing context. A background watcher triggers entries when due, processes them through the agent, and routes responses back.

**Key principle:** All state lives in the file. `cat schedule.jsonl` shows the truth.

## File Format

Location: `workspace/schedule.jsonl`

### One-Shot Entries

Execute once at a specific time, then deleted from file:

```json
{"trigger_at": "2026-01-12T09:00:00Z", "message": "Check the build", "chat_id": "123456", "provider": "telegram", "user_id": "789", "created_at": "2026-01-11T10:00:00Z"}
```

### Periodic Entries

Execute on a cron schedule, `last_run` updated in file after each execution:

```json
{"cron": "0 8 * * *", "message": "Daily summary", "chat_id": "123456", "provider": "telegram", "user_id": "789"}
```

After execution:
```json
{"cron": "0 8 * * *", "message": "Daily summary", "chat_id": "123456", "provider": "telegram", "last_run": "2026-01-12T08:00:00Z"}
```

## Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | No | Stable 8-char hex identifier |
| `message` | string | Yes | Task/message to execute |
| `trigger_at` | ISO 8601 | One-shot | When to trigger (UTC) |
| `cron` | string | Periodic | Cron expression (5-field) |
| `timezone` | string | No | IANA timezone name for cron evaluation |
| `chat_id` | string | Yes | Chat to send response to |
| `chat_title` | string | No | Friendly name for the chat |
| `provider` | string | Yes | Provider name (e.g., "telegram") |
| `user_id` | string | No | User who scheduled the task |
| `username` | string | No | @mention name for responses |
| `created_at` | ISO 8601 | No | When the task was created |
| `last_run` | ISO 8601 | No | Last execution time (periodic only) |

## Cron Format

Standard 5-field cron: `minute hour day month weekday`

Examples:
- `0 8 * * *` - Daily at 8 AM
- `0 9 * * 1` - Mondays at 9 AM
- `*/15 * * * *` - Every 15 minutes
- `0 0 1 * *` - First of each month at midnight

## Agent Usage

The agent uses sandbox CLI commands to create entries:

```bash
# One-time task
ash schedule create "Check the build" --at 2026-01-12T09:00:00Z

# Recurring task
ash schedule create "Daily summary" --cron "0 8 * * *"

# List scheduled tasks
ash schedule list

# Cancel a task by ID
ash schedule cancel --id abc12345
```

The commands automatically inject `chat_id`, `user_id`, and `provider` from environment variables (`ASH_CHAT_ID`, `ASH_USER_ID`, `ASH_PROVIDER`).

**Note:** Scheduling only works from providers with persistent chats (e.g., Telegram). Cannot schedule from CLI.

## Behavior

### One-Shot
1. Agent runs `ash schedule create "msg" --at TIME`
2. Command writes entry with context to `schedule.jsonl`
3. Watcher detects entry is due
4. Handler creates ephemeral session, runs agent with message
5. Response sent back to original chat
6. Entry deleted from file

### Periodic
1. Agent runs `ash schedule create "msg" --cron "EXPR"`
2. Command writes entry with context to `schedule.jsonl`
3. Watcher calculates next run from cron (and `last_run` if present)
4. Handler creates ephemeral session, runs agent with message
5. Response sent back to original chat
6. `last_run` updated in file, entry preserved for next run

## Task Execution Wrapper

When a scheduled task executes, the handler wraps it with timing context so the agent can decide whether the task is still relevant.

### Wrapper Format

The task message is wrapped with XML tags:

- `<context>` - Entry ID, schedule type, scheduled by
- `<timing>` - Current time, scheduled fire time, delay
- `<decision-guidance>` - Rules for skip vs execute
- `<task>` - The original task message

### Time-Sensitive vs Time-Independent

**Time-sensitive tasks** depend on being run close to schedule:
- Greetings tied to time of day ("good morning")
- Reminders for specific moments ("remind me at 2pm")
- Event prompts ("daily standup reminder")

**Time-independent tasks** provide value regardless of delay:
- Data fetching (weather, transit, stocks)
- Reports and summaries
- Backups and syncs

### Skip Decision

The agent uses these thresholds for time-sensitive tasks:
- Delay > 2 hours AND meaning has passed: Skip
- Delay > 4 hours: Almost always skip
- Delay 30 min - 2 hours: Use judgment

Time-independent tasks always execute.

### Output Rules

**If executing:** Run normally, don't mention the delay.

**If skipping:** Brief explanation + next scheduled time if recurring.
Example: "Skipping morning greeting - it's now 3:45 PM. This runs daily at 8 AM."

## Integration

```python
from ash.events import ScheduledTaskHandler, ScheduleWatcher
from pathlib import Path

# Create watcher
watcher = ScheduleWatcher(Path("workspace/schedule.jsonl"))

# Create handler with agent and sender map
handler = ScheduledTaskHandler(
    agent=agent,
    senders={"telegram": telegram_provider.send_message}
)
watcher.add_handler(handler.handle)

# Start watching
await watcher.start()
```

## Verification

```bash
# Start server
uv run ash serve

# In Telegram, tell the bot:
"remind me in 2 minutes to check the build"

# Verify entry was created:
cat workspace/schedule.jsonl

# After 2 minutes, bot sends response to the same chat
# Entry is removed from schedule.jsonl
```

## Design Decisions

1. **Single JSONL file** - Simple, grepable, git-friendly
2. **State in file** - `last_run` persisted, survives restarts
3. **Delete vs update** - One-shot deleted, periodic updated in place
4. **CLI injects context** - `ash schedule create` adds chat_id/provider from env vars
5. **Provider required** - Requires provider with persistent chat for response routing
6. **Fresh context per task** - Each task runs in ephemeral session
7. **UTC times** - Avoids timezone confusion
8. **Ownership filtering** - Users can only see/cancel their own tasks
9. **Time-aware execution** - Agent can skip stale time-sensitive tasks
10. **Timing context** - Handler provides current time, fire time, and delay
