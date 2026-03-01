---
description: "Send, read, search, or draft emails and manage calendar events via Gmail and Google Calendar. Use whenever the user mentions email, inbox, sending a message, checking their calendar, scheduling a meeting, or managing events."
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
      description: The email/calendar task to perform
  required:
    - task
---

Manage Gmail and Google Calendar through host-managed capabilities.

## Security Contract

- Use `ash-sb capability` for every Gmail/Calendar operation.
- Never read or request raw OAuth access tokens, refresh tokens, or client secrets.

## Workflow

On every invocation, follow these steps in order:

### 1. Verify capabilities

```bash
ash-sb capability list
```

If `gog.email` or `gog.calendar` is missing, tell the user that the google skill needs to be enabled in their Ash config (`[skills.google] enabled = true`) and stop.

### 2. Verify authentication

```bash
ash-sb capability auth status --capability gog.email
```

If not authenticated, walk the user through setup:

1. Begin auth:
   ```bash
   ash-sb capability auth begin --capability gog.email --account work
   ```
2. Show the user the `auth_url` and ask them to open it and complete consent.
3. Once the user provides the callback URL or code, complete auth:
   ```bash
   ash-sb capability auth complete --flow-id <flow_id> --code <code>
   ```
4. Confirm success, then proceed to the operation.

Use account hints (`work` / `personal`) only when the user specifies or context is clear.

### 3. Perform operation

Run capability operations with explicit capability IDs and JSON input:

```bash
ash-sb capability invoke --capability gog.email --operation list_messages --input-json '{"folder":"inbox","limit":20}'
ash-sb capability invoke --capability gog.calendar --operation list_events --input-json '{"calendar":"primary","window":"7d"}'
```

Confirm key details before mutating operations (sending emails, creating/updating events).

## Output Format

Format your `complete()` output exactly as shown below. This is critical — the parent agent relays your output directly.

**Listing emails:**

```
- From: Alice <alice@example.com> — "Quarterly review" (2 hours ago)
- From: Bob <bob@example.com> — "Lunch tomorrow?" (yesterday)
```

**Listing events:**

```
- Tomorrow 10am–11am: Team standup (Google Meet)
- Friday 2pm–3pm: 1:1 with Alice (Room 3B)
```

**After mutations (send, create, update, delete):**

```
Sent: "Re: Quarterly review" to alice@example.com
```

```
Created: Team lunch — Friday 12pm–1pm
```

**Formatting rules:**

- Show dates conversationally ("2 hours ago", "tomorrow at 3pm") — never raw ISO timestamps
- After mutations, do NOT re-list all items unless the user asks
- Only claim success after the command produces success output

## Error Handling

- If a command fails, report the error message and stop
- Do not attempt to fix or debug failed commands unless the user asks
