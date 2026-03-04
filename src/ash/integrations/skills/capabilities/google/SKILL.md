---
name: google
description: "Manage Gmail and Google Calendar with capability-backed auth and operations. Use when asked to check inbox, summarize emails, give a day at a glance, send an email, review calendar events, or schedule meetings."
opt_in: true
sensitive: true
access:
  chat_types:
    - private
capabilities:
  - gog.email
  - gog.calendar
allowed_tools:
  - bash
max_iterations: 25
input_schema:
  type: object
  properties:
    task:
      type: string
      description: The Google email/calendar task to perform
  required:
    - task
---

Manage Gmail and Google Calendar through host-managed capabilities.

## Security Contract

- Use `ash-sb capability` for every Gmail/Calendar operation.
- Never read or request raw OAuth access tokens, refresh tokens, or client secrets.
- Do not fabricate capability results. Only report data returned by commands.

## Workflow

On every invocation, follow these steps in order:

### 1. Check capability status

```bash
ash-sb capability list
```

- If a needed capability is missing, tell the user to enable `[skills.google]` and stop.
- If a needed capability is not authenticated, run auth (step 2).
- If already authenticated, continue to operations (step 3).

### 2. Authenticate (when needed)

For each unauthenticated capability (`gog.email`, `gog.calendar`):

```bash
ash-sb capability auth begin -c gog.email
```

Then:

- If flow type is `device_code`: show URL + user code, then poll.
- If flow type is `authorization_code`: show URL and ask user for callback URL or code, then complete.

Use these commands:

```bash
ash-sb capability auth poll --flow-id <id> --timeout 300
ash-sb capability auth complete --flow-id <id> --callback-url '<URL>'
ash-sb capability auth complete --flow-id <id> --code '<CODE>'
```

If user intent is setup-only, stop after successful auth confirmation.

### 3. Perform operations

Use only capability operations and explicit JSON input.

Common email/calendar commands:

```bash
ash-sb capability invoke -c gog.email -o list_messages --input-json '{"folder":"inbox","limit":20}'
ash-sb capability invoke -c gog.email -o search_messages --input-json '{"query":"is:unread newer_than:1d","limit":20}'
ash-sb capability invoke -c gog.email -o get_message --input-json '{"id":"<message_id>"}'
ash-sb capability invoke -c gog.email -o get_thread --input-json '{"thread_id":"<thread_id>","limit":20}'
ash-sb capability invoke -c gog.calendar -o list_events --input-json '{"calendar":"primary","window":"1d"}'
ash-sb capability invoke -c gog.calendar -o create_event --input-json '{"title":"Team sync","start":"2026-03-04T18:00:00Z"}'
```

If the user asks a broad question and does not provide scope, use these defaults:

- Email summaries: `search_messages` with `{"query":"is:unread newer_than:1d","limit":20}`
- Day-at-a-glance: `list_events` with `{"calendar":"primary","window":"1d"}` plus unread/recent email query
- Message deep read: run `get_message` for each item you summarize

## Behavior Playbooks

### Summarize Emails

When user asks for summaries (for example "summarize my emails", "what did I miss"):

1. Gather candidate messages with `search_messages` (preferred) or `list_messages`.
2. Fetch full message content with `get_message` for messages you summarize.
3. Summarize using this structure:
   - `Top priorities`
   - `Needs reply`
   - `FYI`
   - `Suggested next actions`
4. Keep each bullet tied to a concrete message subject/sender so the user can act on it.

Do not summarize from snippets alone when full content can be fetched.

### Day At A Glance

When user asks for a day overview:

1. Pull today/near-term calendar with `list_events`.
2. Pull high-signal recent email using `search_messages` (for example unread/new/important) and fetch full content for top items.
3. Return this structure:
   - `Today's schedule`
   - `Email priorities`
   - `Conflicts / follow-ups`
   - `Recommended next steps`
4. If there are no events or no high-signal email, say that explicitly instead of leaving sections blank.

Use Google calendar + Google email only for this view.

### Standard mutations

Before `send_message` or `create_event`, confirm key details if user intent is ambiguous.
Required confirmation fields:

- Email send: recipient, subject, body intent
- Event create: title, start time/date, end time or duration, timezone context if unclear

## Output Rules

- Keep timestamps conversational (for example "2 hours ago", "tomorrow at 3pm").
- For summary workflows, prefer grouped bullets over raw dumps.
- After mutation success, confirm the action and stop unless user asked for more.
- Only claim success after command output confirms it.
- For auth/setup completion, explicitly state which capability is now connected.

## Error Handling

- If a command fails, report the error message and stop.
- Do not request raw credentials or attempt unsupported workarounds.
- If capability is unavailable or disabled, instruct the user to enable `[skills.google]`.
