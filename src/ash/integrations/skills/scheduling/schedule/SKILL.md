---
description: "Create, list, update, or cancel scheduled tasks and reminders. Use whenever the user asks to be reminded, wants something to happen at a specific time, sets up recurring checks, or asks about upcoming scheduled items."
max_iterations: 10
---

Manage scheduled tasks and reminders using the `ash-sb schedule` CLI.

## Commands

| Command | Description |
|---------|-------------|
| `ash-sb schedule create "msg" --at <time>` | One-time reminder/task (natural language or ISO 8601) |
| `ash-sb schedule create "msg" --cron "<expr>" [--tz TZ]` | Recurring task on a cron schedule |
| `ash-sb schedule list [--all]` | List scheduled tasks (current room by default) |
| `ash-sb schedule cancel --id <id>` | Cancel a scheduled task |
| `ash-sb schedule update --id <id> [--message MSG] [--at TIME] [--cron EXPR] [--tz TZ]` | Update an existing task |

## When to Use `--at` vs `--cron`

| Flag | Use for | Examples |
|------|---------|----------|
| `--at` | One-time reminders and future tasks | "remind me tomorrow at 9am", "check the build in 2 hours" |
| `--cron` | Recurring monitoring, daily summaries, periodic checks | "every weekday at 10am", "every 15 minutes" |

For continuous monitoring, prefer recurring cron checks; use self-rescheduling only when cadence must change dynamically.

## Timezone Handling

- Times default to the user's local timezone. Use `--tz` only when the user specifies a different timezone, including explicit UTC requests.
- If the user specifies a time with an explicit timezone (e.g. "10am ET"), preserve that wall-clock time in that timezone. Example: `10am ET` → `--cron '0 10 * * *' --tz America/New_York` (not a converted hour).
- When a single request includes times in different timezones, create each schedule with its own `--tz` matching the user-specified timezone for that item.

## Task Messages

Write scheduled task messages as self-contained future instructions. The message should make sense when executed later without conversational context.

Good: `"Check the GitHub Actions build status for the main branch and report any failures"`
Bad: `"Check the build"` (ambiguous without context)

## Cron Format

Standard 5-field cron: `minute hour day month weekday`

| Expression | Meaning |
|------------|---------|
| `0 8 * * *` | Daily at 8 AM |
| `0 9 * * 1` | Mondays at 9 AM |
| `0 10 * * 1-5` | Weekdays at 10 AM |
| `*/15 * * * *` | Every 15 minutes |
| `0 0 1 * *` | First of each month at midnight |

## Output Format

Format your `complete()` output exactly as shown below. This is critical — the parent agent relays your output directly.

**After creating a schedule:**

```
Scheduled: Check GitHub Actions build status — tomorrow at 9am
```

```
Scheduled: Daily standup reminder — every weekday at 10am PT
```

**Listing schedules:**

```
- Check GitHub Actions build status — tomorrow at 9am (one-time)
- Daily standup reminder — every weekday at 10am PT (recurring)
```

**After cancel/update:**

```
Cancelled: Daily standup reminder
```

**Formatting rules:**

- Show dates conversationally ("tomorrow at 3pm", "next Monday at 9am") — never raw ISO timestamps
- Hide internal IDs — never show them unless the user asks or a follow-up needs one
- Only claim success after the command produces success output

## Error Handling

- If a command fails, report the error message and stop
- Do not attempt to fix or debug failed commands unless the user asks
- If an ID is needed for cancel/update but not known, list tasks first to find it
