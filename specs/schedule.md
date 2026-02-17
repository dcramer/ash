# Schedule System

File-based task scheduling following the "filesystem first" principle.

## Status: Implemented

## Overview

The schedule system allows the agent to schedule future tasks using sandbox CLI commands (`ash schedule create`). The commands go through RPC to the host process, which writes entries to a JSONL file with routing context. A background watcher triggers entries when due, processes them through the agent, and routes responses back.

**Key principle:** All state lives in the file. `cat schedule.jsonl` shows the truth.

## File Format

Location: `~/.ash/schedule.jsonl`

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

## Sandbox CLI (`ash-sb schedule`)

The agent uses sandbox CLI commands to manage entries. These commands communicate with the host process via RPC (`schedule.create`, `schedule.list`, `schedule.cancel`, `schedule.update`).

The commands automatically inject `chat_id`, `user_id`, `provider`, and `timezone` from environment variables (`ASH_CHAT_ID`, `ASH_USER_ID`, `ASH_PROVIDER`, `ASH_TIMEZONE`).

**Note:** Scheduling only works from providers with persistent chats (e.g., Telegram). Cannot schedule from CLI.

**Design principle:** Output is optimized for LLM consumption — every response includes enough context for the agent to verify correctness and report results to the user. Timezone, next fire time, and task message are always shown.

### `ash-sb schedule create`

```bash
# One-time task (natural language or ISO 8601)
ash-sb schedule create "Check the build" --at "tomorrow at 9am"
ash-sb schedule create "Check the build" --at 2026-01-12T09:00:00Z

# Recurring task
ash-sb schedule create "Daily summary" --cron "0 8 * * *"

# With explicit timezone (overrides ASH_TIMEZONE default)
ash-sb schedule create "Standup" --cron "0 10 * * 1-5" --tz America/New_York
```

**One-shot output:**
```
Scheduled reminder (id=a1b2c3d4)
  Time: Sat 2026-02-21 14:00 (America/Los_Angeles)
  UTC:  2026-02-21T22:00:00Z
  Task: Check the build
```

**Recurring output:**
```
Scheduled recurring task (id=e5f6a7b8)
  Cron: 0 10 * * 1-5 (America/New_York)
  Next: Mon 2026-02-23 10:00
  Task: Standup
```

When timezone is UTC, a hint is shown:
```
  Hint: Use --tz to set timezone (e.g. --tz America/New_York)
```

### `ash-sb schedule list`

```bash
ash-sb schedule list
```

**Output:**
```
Scheduled tasks (times shown in America/Los_Angeles):

  a1b2c3d4  one-shot   Sat 2026-02-21 14:00 (America/Los_Angeles)
           Task: Check the build

  e5f6a7b8  periodic   0 10 * * 1-5 (America/New_York)
           Next: Mon 2026-02-23 10:00
           Task: Standup

Total: 2 task(s)
```

### `ash-sb schedule cancel`

```bash
ash-sb schedule cancel --id a1b2c3d4
```

**Output:**
```
Cancelled task (id=a1b2c3d4): Check the build
```

### `ash-sb schedule update`

```bash
ash-sb schedule update --id a1b2c3d4 --message "New task text"
ash-sb schedule update --id a1b2c3d4 --at "tomorrow at 10am"
ash-sb schedule update --id a1b2c3d4 --cron "0 9 * * *" --tz America/Los_Angeles
```

**Recurring output:**
```
Updated recurring task (id=e5f6a7b8)
  Cron: 0 9 * * * (America/Los_Angeles)
  Next: Mon 2026-02-23 09:00
  Task: Standup
```

**One-shot output:**
```
Updated reminder (id=a1b2c3d4)
  Time: Tue 2026-02-24 10:00 (America/Los_Angeles)
  UTC:  2026-02-24T18:00:00Z
  Task: New task text
```

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

### Update
1. Agent runs `ash schedule update --id ID --message "new text"`
2. RPC validates ownership (user_id must match) and applies changes
3. Cannot switch entry types (one-shot ↔ periodic)
4. `trigger_at` updates must be in the future; `cron` updates must be valid expressions

### No-Retry Semantics

Failed tasks are still marked as processed — one-shot entries are deleted, periodic entries get `last_run` updated. There is no automatic retry. This prevents infinite retry loops for tasks that consistently fail.

### Ownership Rules

- **RPC layer** (sandbox commands): `cancel` and `update` check that the requesting `user_id` matches the entry's `user_id`. Users can only modify their own tasks.
- **CLI commands** (host-side): `ash schedule cancel`, `ash schedule update`, `ash schedule clear` do not check ownership — they are admin commands for the host operator.

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
from ash.scheduling import ScheduleStore, ScheduleWatcher, ScheduledTaskHandler

# Create store and watcher
store = ScheduleStore(get_schedule_file())
watcher = ScheduleWatcher(store, timezone="America/Los_Angeles")

# Create handler with agent, senders, registrars, and executor
handler = ScheduledTaskHandler(
    agent=agent,
    senders={"telegram": telegram_provider.send_message},
    registrars={"telegram": telegram_registrar},
    timezone="America/Los_Angeles",
    agent_executor=agent_executor,
)
watcher.add_handler(handler.handle)

# Wire store into RPC for sandbox commands
register_schedule_methods(rpc_server, store)

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
cat ~/.ash/schedule.jsonl

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
8. **Ownership filtering** - Users can only see/cancel their own tasks (RPC layer)
9. **Time-aware execution** - Agent can skip stale time-sensitive tasks
10. **Timing context** - Handler provides current time, fire time, and delay
11. **No retry** - Failed tasks are marked processed to prevent infinite loops
