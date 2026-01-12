# Schedule System

File-based task scheduling following the "filesystem first" principle.

## Status: Implemented

## Overview

The schedule system allows the agent to schedule future tasks using the `schedule_task` tool. The tool writes entries to a JSONL file with routing context. A background watcher triggers entries when due, processes them through the agent, and routes responses back.

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
| `message` | string | Yes | Task/message to execute |
| `trigger_at` | ISO 8601 | One-shot | When to trigger (UTC) |
| `cron` | string | Periodic | Cron expression (5-field) |
| `chat_id` | string | Yes | Chat to send response to |
| `provider` | string | Yes | Provider name (e.g., "telegram") |
| `user_id` | string | No | User who scheduled the task |
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

The agent uses the `schedule_task` tool to create entries:

```
# One-time task
schedule_task(message="Check the build", trigger_at="2026-01-12T09:00:00Z")

# Recurring task
schedule_task(message="Daily summary", cron="0 8 * * *")
```

The tool automatically injects `chat_id`, `user_id`, and `provider` from the current context.

**Note:** Scheduling only works from providers with persistent chats (e.g., Telegram). Cannot schedule from CLI.

## Behavior

### One-Shot
1. Agent calls `schedule_task` with `trigger_at`
2. Tool writes entry with context to `schedule.jsonl`
3. Watcher detects entry is due
4. Handler creates ephemeral session, runs agent with message
5. Response sent back to original chat
6. Entry deleted from file

### Periodic
1. Agent calls `schedule_task` with `cron`
2. Tool writes entry with context to `schedule.jsonl`
3. Watcher calculates next run from cron (and `last_run` if present)
4. Handler creates ephemeral session, runs agent with message
5. Response sent back to original chat
6. `last_run` updated in file, entry preserved for next run

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
4. **Tool injects context** - `schedule_task` adds chat_id/provider automatically
5. **CLI not supported** - Requires provider with persistent chat for response routing
6. **Fresh context per task** - Each task runs in ephemeral session
7. **UTC times** - Avoids timezone confusion
