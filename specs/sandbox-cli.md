# Sandbox CLI (`ash-sb`)

> Agent-facing command interface inside the Docker sandbox

Files: packages/ash-sandbox-cli/

## Intent

The sandbox CLI is the agent's primary interface for interacting with Ash subsystems. Every command runs inside the sandbox container, communicates with the host via RPC, and produces output that the LLM agent reads to decide what to do next.

**The user of this CLI is an LLM, not a human.** Design every output for machine consumption first.

## Output Design Principles

### 1. Every response must be self-verifying

The agent should be able to confirm correctness from the output alone, without making a follow-up call.

```
# BAD - agent can't verify what was scheduled or when
Scheduled task abc12345

# GOOD - agent can verify ID, time, timezone, and content
Scheduled reminder (id=abc12345)
  Time: Sat 2026-02-21 14:00 (America/New_York)
  UTC:  2026-02-21T19:00:00Z
  Task: Check the build
```

### 2. Always echo back what was stored

Don't just confirm "done" — show the agent what the system actually persisted. This catches misunderstandings between the agent's intent and the system's interpretation.

- Times: show both local and UTC
- Cron: show the expression, timezone, and next fire time
- Text: show a preview of the stored message
- IDs: always include them for follow-up operations

### 3. Include timezone in every time-related output

The agent has no inherent timezone awareness. Every time shown must include its timezone context.

```
# BAD - 14:00 in what timezone?
Time: 2026-02-21 14:00

# GOOD
Time: Sat 2026-02-21 14:00 (America/New_York)
```

### 4. Provide hints when the agent likely made a mistake

When the output suggests the agent may have used incorrect defaults, add a hint line.

```
Scheduled recurring task (id=e5f6a7b8)
  Cron: 0 10 * * 1-5 (UTC)
  Task: Daily standup
  Hint: Use --tz to set timezone (e.g. --tz America/New_York)
```

### 5. Error messages should suggest the fix

Don't just say what went wrong — say how to fix it.

```
# BAD
Error: Invalid time

# GOOD
Error: Could not parse time: 'next tuesday 3pm EST'
```

```
# BAD
Error: Missing option

# GOOD
Error: Must specify either --at (one-time) or --cron (recurring)
```

### 6. Use structured, line-oriented output

The agent parses output line by line. Use consistent prefixed fields rather than dense tables or prose.

```
# Preferred: labeled fields
Scheduled reminder (id=abc12345)
  Time: Sat 2026-02-21 14:00 (America/New_York)
  UTC:  2026-02-21T19:00:00Z
  Task: Check the build

# Avoid: dense tables that are hard to parse
ID        Type      Schedule          Message
abc12345  one-shot  2026-02-21 14:00  Check the build
```

For list commands, use one block per entry with labeled fields rather than columnar tables.

### 7. Include counts in list output

Always end list commands with a total count. This lets the agent verify it got all results.

```
Total: 3 task(s)
```

## Commands

| Command Group | Purpose |
|---------------|---------|
| `browser` | Start/list/control browser sessions and page actions |
| `schedule` | Create, list, cancel, update scheduled tasks |
| `todo` | Manage canonical todo items and reminder links |
| `memory` | Search, list, add, extract, delete memories |
| `logs` | Query structured server logs |
| `config` | Reload configuration |
| `skill` | Validate and list workspace skills |

See individual subsystem specs for command-specific output formats:
- Schedule: [specs/schedule.md](schedule.md)
- Browser: [docs/src/content/docs/systems/browser.mdx](../docs/src/content/docs/systems/browser.mdx)

## Environment Variables

The sandbox receives these from the host via container environment:

| Variable | Source | Purpose |
|----------|--------|---------|
| `ASH_SESSION_ID` | Current session | Session context |
| `ASH_USER_ID` | Message sender | Ownership checks |
| `ASH_CHAT_ID` | Current chat | Response routing |
| `ASH_CHAT_TITLE` | Current chat | Display name |
| `ASH_PROVIDER` | Current provider | Response routing |
| `ASH_USERNAME` | Message sender | @mention name |
| `ASH_DISPLAY_NAME` | Message sender | Human name |
| `ASH_TIMEZONE` | Config | Default timezone for times |
| `ASH_MESSAGE_ID` | Current message | Message context |
| `ASH_RPC_SOCKET` | Host RPC | Unix socket path |
| `ASH_MOUNT_PREFIX` | Config | Path prefix for mounts |

## Architecture

```
Agent → bash tool → ash-sb command → RPC call → Host process → Subsystem
                                   ↓
                              stdout/stderr → Agent reads output
```

All `ash-sb` commands are thin wrappers that:
1. Read routing context from environment variables
2. Validate inputs locally (time parsing, cron validation)
3. Call the host via RPC over Unix socket
4. Format the RPC response for agent consumption

No business logic lives in the sandbox CLI. It's a presentation layer.

## Verification

For each command:
- [ ] Success output includes all stored fields
- [ ] Every time includes timezone
- [ ] IDs are shown for follow-up operations
- [ ] Error messages suggest the fix
- [ ] List commands end with count
- [ ] Agent can verify correctness without a follow-up call
