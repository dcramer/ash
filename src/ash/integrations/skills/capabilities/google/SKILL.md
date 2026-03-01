---
description: "Set up, configure, or use Gmail and Google Calendar. Handles initial setup, authentication, sending/reading/searching emails, and managing calendar events. Use whenever the user mentions setting up Google, email, inbox, sending a message, checking their calendar, scheduling a meeting, or managing events."
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

Try a lightweight read operation to check if auth is in place:

```bash
ash-sb capability invoke -c gog.email -o list_messages --input-json '{"limit":1}'
```

If the output contains `capability_auth_required`, the user needs to authenticate.
Walk them through setup:

1. Begin auth (use `--account work` or `--account personal` if the user specifies):
   ```bash
   ash-sb capability auth begin -c gog.email
   ```
2. Show the user the `auth_url` from the output and ask them to open it and complete consent.
3. Once the user provides the callback URL or authorization code, complete auth:
   ```bash
   ash-sb capability auth complete --flow-id <FLOW_ID> --code <CODE>
   ```
   Or with callback URL:
   ```bash
   ash-sb capability auth complete --flow-id <FLOW_ID> --callback-url <URL>
   ```
4. Confirm success, then proceed to the original operation.

If both `gog.email` and `gog.calendar` are needed, authenticate each separately.

### 3. Perform operation

Run capability operations with explicit capability IDs and JSON input:

```bash
ash-sb capability invoke -c gog.email -o list_messages --input-json '{"folder":"inbox","limit":20}'
ash-sb capability invoke -c gog.calendar -o list_events --input-json '{"calendar":"primary","window":"7d"}'
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
