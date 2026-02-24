# Capabilities

> Token-authenticated host capability API for sensitive external systems used by skills.

Files: `src/ash/capabilities/`, `src/ash/rpc/methods/capability.py`, `src/ash/integrations/capabilities.py`, `packages/ash-sandbox-cli/src/ash_sandbox_cli/commands/capability.py`

## Status

Contract drafted. Implementation pending.

## Intent

Capabilities provide a secure interface for skills to access sensitive external systems
(for example email/calendar) without giving skills direct credentials.

This is the core contract for external skill repos (such as `gogcli`) so they can
share skill code while preserving per-user data and credential isolation.

## Outcomes

1. Skills invoke host-managed capabilities instead of using long-lived API secrets in env vars.
2. Verified caller context (`ASH_CONTEXT_TOKEN`) determines identity/scope for all capability calls.
3. Multiple users can invoke the same skill implementation without cross-user data exposure.
4. Sensitive capability operations are restricted by chat policy (DM-only by default).
5. Auth flows (OAuth/device/browser) are mediated by host APIs and short-lived flow handles.

## Requirements

### MUST

- All capability RPC methods require valid `context_token` and fail closed otherwise.
- Identity/routing for capability execution must come only from verified token claims.
- Caller-provided identity/routing fields (`user_id`, `chat_id`, `chat_type`, etc.) are not trusted.
- Bot-initiated capability operations must run through `ash-sb` commands (no direct provider/chat path to credential lookups).
- Skills must not receive capability credentials via environment variables.
- Credential/materialized account state is isolated by verified `effective_user_id` (`sub` claim).
- Capability definitions include access metadata (sensitivity + allowed chat types).
- `sensitive` capabilities default to `private` chat type unless explicitly overridden.
- Capability auth flow handles are short-lived, unguessable, and bound to the requesting user scope.
- Capability execution emits structured audit events without logging raw bearer tokens.
- Capability responses must never include raw credential artifacts (access tokens, refresh tokens, cookie jars, client secrets).

### SHOULD

- Mutating operations support optional idempotency keys.
- Capability-sidecar/runtime bridges reuse authenticated loopback bearer-token patterns.
- Capability stores separate credential material from operation data/artifacts.
- Flow completion should support callback URL ingestion and manual code fallback.

### MAY

- Capability-level rate limits and quotas.
- Explicit shared-account delegation across users/chats (opt-in policy only).

## Trust Model

`ASH_CONTEXT_TOKEN` establishes *who/where* a call originates. Capabilities add policy
and data isolation on top of that:

1. Verify token and extract trusted claims (`sub`, `chat_id`, `chat_type`, `thread_id`, etc.).
2. Evaluate access policy (skill access + capability access + chat policy).
3. Resolve user-scoped credential/account context.
4. Execute provider operation in host-managed boundary.
5. Return sanitized result to sandbox caller.

The token is necessary but not sufficient; capability-specific auth and isolation are
required for secure multi-user operation.

## Access Path Invariant

For agent/bot execution, capability access is constrained to:

`bot tool call -> bash -> ash-sb capability -> RPC (context_token) -> host capability manager`

Important implications:

- `ash-sb` auto-attaches trusted caller context (`ASH_CONTEXT_TOKEN`) to RPC requests.
- Host capability manager resolves credential scope from verified token claims.
- Skills and prompt text cannot select another user's credential scope by passing `user_id`.
- Host/provider-side credential APIs are not exposed directly to arbitrary chat interactions.

## Interface

### Capability Definition

```python
@dataclass
class CapabilityDefinition:
    id: str  # e.g. "gog.email"
    description: str
    sensitive: bool = False
    allowed_chat_types: list[str] = field(default_factory=list)  # empty => all
    operations: dict[str, CapabilityOperation] = field(default_factory=dict)


@dataclass
class CapabilityOperation:
    name: str
    description: str
    requires_auth: bool = True
    mutating: bool = False
    input_schema: dict[str, Any] = field(default_factory=dict)   # JSON Schema
    output_schema: dict[str, Any] = field(default_factory=dict)  # JSON Schema
```

### RPC Methods

#### `capability.list`

Returns capabilities visible to the verified caller context.

Request params:

```json
{
  "include_unavailable": false,
  "context_token": "<signed-token>"
}
```

Response:

```json
{
  "capabilities": [
    {
      "id": "gog.email",
      "description": "Email operations",
      "available": true,
      "requires_auth": true
    }
  ]
}
```

#### `capability.invoke`

Executes one operation under verified caller scope.

Request params:

```json
{
  "capability": "gog.email",
  "operation": "list_messages",
  "input": {"folder": "inbox", "limit": 20},
  "idempotency_key": "optional-client-key",
  "context_token": "<signed-token>"
}
```

Response:

```json
{
  "ok": true,
  "output": {"messages": []},
  "request_id": "cap_01..."
}
```

#### `capability.auth.begin`

Starts auth for a capability/account and returns an auth flow handle.

Request params:

```json
{
  "capability": "gog.email",
  "account_hint": "work",
  "context_token": "<signed-token>"
}
```

Response:

```json
{
  "flow_id": "caf_01...",
  "auth_url": "https://...",
  "expires_at": "2026-02-24T20:10:00Z"
}
```

#### `capability.auth.complete`

Completes a pending auth flow with callback URL or code.

Request params:

```json
{
  "flow_id": "caf_01...",
  "callback_url": "https://localhost/callback?code=...",
  "context_token": "<signed-token>"
}
```

Response:

```json
{
  "ok": true,
  "account_ref": "acct_work"
}
```

### Sandbox CLI Contract (`ash-sb capability`)

- `ash-sb capability list`
- `ash-sb capability invoke --capability <id> --operation <name> --input-json <json>`
- `ash-sb capability auth begin --capability <id> [--account <hint>]`
- `ash-sb capability auth complete --flow-id <id> (--callback-url <url> | --code <code>)`

All commands must use the same `ASH_CONTEXT_TOKEN` trust chain as other sandbox CLI
commands. No direct credential env vars are a supported auth path.
The command layer carries identity context; the host resolves credentials internally.

## Policy Layering

Capability invocation is allowed only when all policy gates pass:

1. Skill-level access policy (for example `sensitive`, `access.chat_types`, `allow_chat_ids`).
2. Capability-level access policy (sensitivity/chat-type constraints).
3. Capability operation preconditions (auth present, required inputs, provider health).

A deny at any layer must fail closed with a deterministic error.

## Data Isolation

Capability state is scoped to verified identity and namespace:

- Credential key space: `(effective_user_id, capability_id, account_ref)`
- Operation state/artifacts: at least `(effective_user_id, capability_id)`
- Optional chat-scoped data: additionally keyed by verified `chat_id`

Cross-user reads/writes are always denied unless an explicit sharing policy exists.

## Browser/Auth Bridge Unification

When capability auth/execution needs sidecar processes, use the same security model as
the browser bridge:

- Loopback-only bridge
- Bearer-token authentication
- Scope-keyed runtime/container identity
- No unauthenticated control channel

See `specs/browser.md` for runtime bridge invariants.

## Errors

| Condition | Error |
|-----------|-------|
| Capability not found | `capability_not_found` |
| Access denied by chat policy | `capability_access_denied` |
| Auth required but missing | `capability_auth_required` |
| Auth flow expired/invalid | `capability_auth_flow_invalid` |
| Invalid input schema | `capability_invalid_input` |
| Upstream/provider unavailable | `capability_backend_unavailable` |

## Verification

- Unit tests for capability manager auth/policy enforcement.
- RPC tests proving caller identity fields are token-derived only.
- Sandbox CLI tests for `ash-sb capability` command contracts.
- Integration tests for multi-user isolation (same skill, different users).
- Integration tests for sensitive capability DM-only default behavior.
