# Capability Auth Flows

> Auth flow strategies for headless capability providers, starting with Device Authorization Grant (RFC 8628).

Files: `src/ash/capabilities/providers/base.py`, `src/ash/capabilities/manager.py`, `src/ash/capabilities/providers/subprocess.py`, `src/ash/rpc/methods/capability.py`, `packages/ash-sandbox-cli/src/ash_sandbox_cli/commands/capability.py`

Parent spec: `specs/capabilities.md`

## Status

Spec only — not yet implemented.

## Intent

Ash runs headless on remote machines. Users interact via Telegram or CLI over SSH.
Standard redirect-based OAuth flows break because the bot cannot open a browser or
receive HTTP callbacks.

This spec defines a universal auth strategy — **Device Authorization Grant
(RFC 8628)** — that works through chat messages alone:

1. Bot shows user a URL + short code.
2. User visits URL on their own device, enters code, approves.
3. Bot detects approval via polling.

The `provider` field in `CapabilityCallContext` carries `"cli"`, `"telegram"`, etc.
The device code flow works identically across all providers.

## Outcomes

1. Auth flows work on any deployment: headless server, SSH session, Telegram bot.
2. No browser or public URL is required on the machine running Ash.
3. Users complete consent on their own device — phone, laptop, or any browser.
4. Existing `auth_begin`/`auth_complete` (authorization code) flows continue to work unchanged.
5. Skills detect flow type from `auth_begin` response and adapt UX accordingly.

## Device Code Flow

### End-to-end sequence

1. Skill detects `capability_auth_required` from a failed invoke.
2. Skill calls `ash-sb capability auth begin -c gog.email`.
3. Bridge requests device code from provider (e.g. `POST https://oauth2.googleapis.com/device/code`).
4. Bridge returns: `auth_url` (verification URL), `user_code`, `flow_type: "device_code"`, `poll_interval_seconds`.
5. Skill tells user in chat: _"Go to **google.com/device** and enter code **ABCD-EFGH**"_
6. Skill calls `ash-sb capability auth poll --flow-id <id> --timeout 300`.
7. Bridge polls provider's token endpoint at the specified interval.
8. User completes consent on their device.
9. Poll returns success, credentials stored in vault.
10. Skill proceeds with the original operation.

### Flow type detection

`auth_begin` returns a `flow_type` field:

- `"authorization_code"` — existing redirect-based flow (default, backward compatible).
- `"device_code"` — device authorization grant; `user_code` and `poll_interval_seconds` are present.

Skills branch on `flow_type` to determine UX:

```
flow_type == "device_code":
  → Show verification URL + user code, then poll
flow_type == "authorization_code":
  → Show auth URL, ask user to paste callback/code (current behavior)
```

## Contract Changes

### `CapabilityAuthBeginResult`

Add fields to `src/ash/capabilities/providers/base.py`:

```python
@dataclass(slots=True)
class CapabilityAuthBeginResult:
    """Provider response for auth flow initialization."""

    auth_url: str
    flow_type: str = "authorization_code"   # or "device_code"
    user_code: str | None = None            # device code flow only
    poll_interval_seconds: int | None = None  # device code flow only
    expires_at: datetime | None = None
    flow_state: dict[str, Any] = field(default_factory=dict)
```

- `flow_type` defaults to `"authorization_code"` for backward compatibility.
- `user_code` is the short code the user enters on the verification page.
- `poll_interval_seconds` is the minimum polling interval from the provider.

### New `CapabilityAuthPollResult`

Add to `src/ash/capabilities/providers/base.py`:

```python
@dataclass(slots=True)
class CapabilityAuthPollResult:
    """Provider response for device code auth polling."""

    status: str  # "pending" | "complete"
    retry_after_seconds: int | None = None
    account_ref: str | None = None
    credential_material: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

### `CapabilityProvider` protocol

Add `auth_poll` to `src/ash/capabilities/providers/base.py`:

```python
async def auth_poll(
    self,
    *,
    capability_id: str,
    flow_state: dict[str, Any],
    context: CapabilityCallContext,
) -> CapabilityAuthPollResult: ...
```

The `CapabilityManager._provider_auth_poll` wrapper handles the unsupported case:
when `provider_impl is None` or the provider does not implement `auth_poll`, raise
`CapabilityError("capability_invalid_input", "auth polling not supported by this provider")`.
This matches the existing delegation pattern (e.g. `_provider_auth_begin`).

### `CapabilityManager`

Add `auth_poll` method to `src/ash/capabilities/manager.py`:

- Validate flow exists and belongs to user (same checks as `auth_complete`).
- Reject polls on non-device flows (`flow_type != "device_code"` → `capability_invalid_input`).
- Delegate to provider's `auth_poll`.
- On `"complete"`: store account/credentials (same path as `auth_complete`), delete flow.
- On `"pending"`: return retry info to caller.

`CapabilityAuthFlow` (types.py) gains a `flow_type` field:

```python
@dataclass(slots=True)
class CapabilityAuthFlow:
    flow_id: str
    capability_id: str
    user_id: str
    account_hint: str | None
    expires_at: datetime
    flow_type: str = "authorization_code"  # or "device_code"
    flow_state: dict[str, Any] = field(default_factory=dict)
```

`auth_begin` stores `flow_type` from the provider result. `auth_poll` checks
`flow.flow_type == "device_code"` before delegating.

`auth_begin` return dict includes new fields:

```python
return {
    "flow_id": flow_id,
    "auth_url": begin_result.auth_url,
    "flow_type": begin_result.flow_type,
    "user_code": begin_result.user_code,
    "poll_interval_seconds": begin_result.poll_interval_seconds,
    "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
}
```

### `SubprocessCapabilityProvider`

Add `auth_poll` bridge dispatch to `src/ash/capabilities/providers/subprocess.py`:

```python
async def auth_poll(
    self,
    *,
    capability_id: str,
    flow_state: dict[str, Any],
    context: CapabilityCallContext,
) -> CapabilityAuthPollResult:
    result = await self._call_bridge(
        "auth_poll",
        {
            "capability_id": capability_id,
            "flow_state": dict(flow_state),
            "context_token": self._issue_context_token(context),
        },
    )
    status = _required_text(
        value=result.get("status"),
        code="capability_invalid_output",
        message="bridge auth_poll must return status",
    )
    return CapabilityAuthPollResult(
        status=status,
        retry_after_seconds=result.get("retry_after_seconds"),
        account_ref=result.get("account_ref"),
        credential_material=_as_object(result.get("credential_material"), default={}),
        metadata=_as_object(result.get("metadata"), default={}),
    )
```

`auth_begin` response parsing also extracts `flow_type`, `user_code`, `poll_interval_seconds`.

### Bridge-v1 protocol

New `auth_poll` method:

Request:

```json
{
  "version": 1,
  "id": "cap_bridge_abc123",
  "namespace": "gog",
  "method": "auth_poll",
  "params": {
    "capability_id": "gog.email",
    "flow_state": {},
    "context_token": "<signed-token>"
  }
}
```

Pending response:

```json
{
  "version": 1,
  "id": "cap_bridge_abc123",
  "result": {
    "status": "pending",
    "retry_after_seconds": 5
  }
}
```

Complete response:

```json
{
  "version": 1,
  "id": "cap_bridge_abc123",
  "result": {
    "status": "complete",
    "account_ref": "default",
    "credential_material": {"credential_key": "cred_..."},
    "metadata": {}
  }
}
```

### RPC

Register `capability.auth.poll` in `src/ash/rpc/methods/capability.py`:

Request params:

```json
{
  "flow_id": "caf_01...",
  "context_token": "<signed-token>"
}
```

Response (pending):

```json
{
  "status": "pending",
  "retry_after_seconds": 5
}
```

Response (complete):

```json
{
  "ok": true,
  "account_ref": "default"
}
```

### Sandbox CLI

Add `auth poll` subcommand to `packages/ash-sandbox-cli/.../commands/capability.py`:

```
ash-sb capability auth poll --flow-id <id> [--timeout <secs>] [--interval <secs>]
```

- `--flow-id` (required): auth flow handle from `auth begin`.
- `--timeout` (optional): blocking mode — polls repeatedly until success, expiry, or timeout (seconds). The skill agent calls this once and waits.
- `--interval` (optional): override poll interval in seconds (defaults to server-reported `retry_after_seconds` or 5s).

Without `--timeout`, performs a single poll and returns immediately.

## Google OAuth Specifics

- **Client type**: "TVs and Limited Input devices" in Google Cloud Console.
- **Scopes**: `gmail.readonly`, `gmail.send`, `calendar.readonly`, `calendar.events`.
- **Device code endpoint**: `POST https://oauth2.googleapis.com/device/code`.
- **Token endpoint**: `POST https://oauth2.googleapis.com/token` with `grant_type=urn:ietf:params:oauth:grant-type:device_code`.
- **Token refresh**: bridge refreshes expired access tokens before invoke operations.
- **Storage**: access_token + refresh_token in vault via `FileVault.put_json`, keyed by `(user_id, capability_id, account_ref)`.
- **Config**: `google_client_id` / `google_client_secret` in `[skills.google]`, passed to bridge via env vars.

## SKILL.md Update

Update `src/ash/integrations/skills/capabilities/google/SKILL.md` auth section to handle device code flow:

```
1. ash-sb capability auth begin -c gog.email
2. If flow_type is "device_code":
   - Show user: "Visit <verification_url> and enter code: <user_code>"
   - Run: ash-sb capability auth poll --flow-id <id> --timeout 300
   - On success, proceed
3. If flow_type is "authorization_code" (fallback):
   - Show user the auth_url, ask for code paste (current behavior)
```

## Backward Compatibility

- `flow_type` defaults to `"authorization_code"` — existing bridges and skill flows keep working.
- `auth_poll` is handled by the manager delegation wrapper: when the provider is missing or doesn't implement it, raises `CapabilityError("capability_invalid_input", "auth polling not supported by this provider")`.
- Existing `auth_complete` with `code`/`callback_url` continues to work for redirect-based flows.
- `SubprocessCapabilityProvider` handles missing `auth_poll` from bridges gracefully (bridge returns error, provider surfaces it).
- New fields in `auth_begin` return dict (`flow_type`, `user_code`, `poll_interval_seconds`) are nullable/defaulted — callers that don't check them are unaffected.

## Errors

| Condition | Error code |
|-----------|-----------|
| Poll on non-device flow | `capability_invalid_input` |
| Device flow expired | `capability_auth_flow_expired` |
| Device flow denied by user | `capability_auth_flow_denied` |
| Provider doesn't support poll | `capability_invalid_input` |
| Flow not found / wrong user | `capability_auth_flow_invalid` |

## Verification

- Unit tests for `CapabilityManager.auth_poll` lifecycle (pending → complete → account stored).
- Unit tests for `auth_poll` on expired/invalid/wrong-user flows.
- Unit tests for `auth_begin` returning device code fields.
- Unit tests for `SubprocessCapabilityProvider.auth_poll` bridge dispatch.
- RPC tests for `capability.auth.poll` method registration and param validation.
- Sandbox CLI tests for `ash-sb capability auth poll` command contract.
- Backward-compat tests: `auth_begin`/`auth_complete` without device code fields still works.
- Integration tests: full device code flow end-to-end with mock bridge.
