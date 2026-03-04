# Auth and Failure Handling

Use this reference when capability auth is missing, expired, or command execution fails.

## Auth Flow Handling

1. Run `ash-sb capability list`.
2. For any required capability with `Authenticated: no`, run `auth begin`.
3. Complete with the appropriate flow:
   - `device_code`: show URL + code, then poll.
   - `authorization_code`: ask for callback URL or code, then `auth complete`.

If user intent is setup-only, stop after auth succeeds.

## Callback URL vs Code

If user provides a callback URL (`http://localhost/?code=...`), pass it with `--callback-url`.
If user provides only a code value, pass it with `--code`.

Never ask for OAuth secrets.

## Expired/Invalid Flow Recovery

If `auth complete` reports invalid/expired flow:

1. Tell the user their previous auth flow expired or no longer matches active state.
2. Start a fresh auth flow.
3. Ask them to complete promptly and provide the new callback URL/code.

## Capability Availability Errors

If capability is missing entirely:

- Tell user to enable `[skills.google]`.
- Stop rather than guessing alternatives.

## Operation Failures

On failure:

1. Report the command error clearly.
2. Do not claim success.
3. Do not invent fallback data.
4. Stop unless user asks for troubleshooting.

## Mutating Action Safety

Before mutating operations:

- `send_message`: confirm recipient, subject intent, and body intent when ambiguous.
- `create_event`: confirm title, date/time, and duration/end time when ambiguous.
