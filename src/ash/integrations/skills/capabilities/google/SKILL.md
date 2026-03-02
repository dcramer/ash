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

### 1. Check capability status

```bash
ash-sb capability list
```

Expected output:

```
Capabilities:
- gog.email: Gmail integration
  Available: yes
  Authenticated: no
  Operations: list_messages, get_message, send_message, ...
- gog.calendar: Google Calendar integration
  Available: yes
  Authenticated: yes
  Operations: list_events, create_event, ...
Total: 2 capability(ies)
```

- If a needed capability is missing entirely, tell the user the google skill needs to be enabled in their Ash config (`[skills.google] enabled = true`) and stop.
- If `Authenticated: no` for any needed capability, go to step 2.
- If `Authenticated: yes` for all needed capabilities, skip to step 3.

### 2. Authenticate

Run this step for each capability where `Authenticated: no`. If the user's request is setup-only (e.g. "set up my email"), stop after authentication is complete — do not invoke any operations.

**2a. Begin auth flow**

Use `--account work` or `--account personal` if the user specifies an account preference:

```bash
ash-sb capability auth begin -c gog.email
```

Expected output:

```
Started capability auth flow (flow_id=abc123)
  Capability: gog.email
  Auth URL: https://www.google.com/device
  Flow type: device_code
  User code: ABCD-EFGH
  Poll interval: 5s
  Expires: 2026-03-01T12:30:00Z
```

**2b. Present URL and code to user**

Check the `Flow type` in the output:

- If `device_code`: show the `Auth URL` and `User code` from the output. Tell the user to open the URL and enter the code. Then proceed to step 2c to poll for completion.
- If `authorization_code`: show the `Auth URL` from the output and ask the user to complete the Google consent screen and provide either the authorization code or the callback URL. Then use `ash-sb capability auth complete --flow-id <id> --code <CODE>`.

**2c. Poll for completion (device code flow)**

After showing the user the URL and code, poll for completion:

```bash
ash-sb capability auth poll --flow-id abc123 --timeout 300
```

This blocks until the user completes authorization or the timeout expires. Expected output on success:

```
Capability auth completed (flow_id=abc123, account_ref=default)
```

**2d. Repeat for additional capabilities**

If multiple capabilities need auth (e.g. both `gog.email` and `gog.calendar`), repeat steps 2a–2c for each one. Each capability requires its own auth flow.

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

**Setup/auth completion:**

```
Google email is now connected (user@gmail.com). You're all set — let me know when you want to check your inbox or send a message.
```

```
Google email and calendar are now connected. You're all set to use both.
```

**Formatting rules:**

- Show dates conversationally ("2 hours ago", "tomorrow at 3pm") — never raw ISO timestamps
- After mutations, do NOT re-list all items unless the user asks
- Only claim success after the command produces success output

## Error Handling

- If a command fails, report the error message and stop
- Do not attempt to fix or debug failed commands unless the user asks
