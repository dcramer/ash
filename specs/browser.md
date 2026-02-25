# Browser Subsystem

## Scope

Browser provides session-scoped page automation and artifacts for agent/tool workflows.

It owns:
- Browser session lifecycle (`start`, `list`, `show`, `close`, `archive`)
- Deterministic page actions (`goto`, `extract`, `click`, `type`, `wait_for`, `screenshot`)
- Browser artifacts/state under `~/.ash/browser/`

It does not own:
- Message transport
- LLM orchestration
- Agent runtime policy

## Providers

- `sandbox`: Browser automation MUST execute inside the sandbox/container runtime.
- `kernel`: Remote provider adapter.

### Sandbox Runtime Controls

`[browser.sandbox]` supports:

- `runtime_required` (bool): deprecated compatibility flag; sandbox-only runtime is enforced regardless of value.
- `runtime_warmup_on_start` (bool): run startup warmup hook during integration startup. For scope-keyed dedicated runtime, this performs preflight checks without spawning a default scope container.
- `runtime_restart_attempts` (int): bounded restart attempts when runtime becomes unhealthy.
- `container_image` (string): dedicated browser image used by the sandbox provider.
- `container_name_prefix` (string): prefix for scope-hashed dedicated container names.

## Hard Requirements

1. `sandbox` provider MUST NOT execute browser automation directly in host runtime.
2. `sandbox` provider MUST use sandbox/container runtime dependencies (Chromium/Playwright).
3. Provider behavior and operational docs MUST make the runtime boundary explicit.
4. Session routing MUST NOT silently cross providers.
5. Retention settings MUST be enforced at runtime:
   - `browser.max_session_minutes`
   - `browser.artifacts_retention_days`
6. `sandbox` provider SHOULD keep a warm Chromium runtime per sandbox container and attach sessions to it.
7. `sandbox` provider startup MUST gate on CDP readiness in two phases:
   - HTTP endpoint readiness (`/json/version` with `webSocketDebuggerUrl`)
   - CDP websocket handshake readiness
8. CDP startup failures MUST include actionable diagnostics (phase + process/log details).
9. Runtime/doctor guidance MUST NOT document host-runtime bypasses for sandbox provider.
10. Sandbox browser runtime MUST be dedicated-container only; legacy shared-executor Chromium launch paths are unsupported.
11. Dedicated browser containers MUST be scope-keyed (effective user scope) so independent scopes do not share a runtime container.
12. Dedicated runtime command execution MUST traverse an authenticated loopback bridge using short-lived signed tokens with scope/target claims (not static bearer secrets).
13. Long-lived dedicated runtimes MUST mint bridge tokens per request (or equivalent bounded rotation) so token expiry does not break healthy sessions.
14. Integration shutdown MUST perform best-effort browser runtime teardown so service restarts do not leave dangling dedicated browser containers.

## Integration Contract

- Browser registers through integration composition hooks (`specs/subsystems.md` + `specs/integrations.md`).
- Browser RPC and tool surfaces MUST delegate to the browser subsystem manager.
- Core harness MUST NOT add browser-specific orchestration branches outside integration hooks.

## Public Action Contract

Manager action surface:

- Session lifecycle:
  - `session.start`
  - `session.list`
  - `session.show`
  - `session.close`
  - `session.archive`
- Page actions:
  - `page.goto`
  - `page.extract`
  - `page.click`
  - `page.type`
  - `page.wait_for`
  - `page.screenshot`

Required behavior:

1. Session routing must be explicit by `session_id` or `session_name`.
2. Cross-provider session fallback is rejected.
3. `page.goto` writes latest HTML/title metadata for follow-up extraction.
4. `page.extract` supports `mode=text|title`.
5. `page.screenshot` writes artifact refs under browser artifact storage.
6. Error responses are normalized in `BrowserActionResult` and include actionable runtime hints when browser runtime is unavailable.

## Security Invariants

1. Browser sandbox provider never executes browser automation in host runtime.
2. Browser manager runtime gate must fail closed when sandbox runtime is unavailable.
3. Browser provider implementations must execute via sandbox executor/runtime process.
4. Tool/rpc/cli surfaces must all route through manager/runtime gate (single enforcement point).

## Operational Verification

CLI smoke path:

- `ash browser smoke <url>`
- Flow: `session.start -> page.goto -> page.extract(title/text) -> page.screenshot -> session.archive`
- This command is required for rapid runtime validation in production-like environments.

Expected failure semantics:

- Runtime unavailable: `sandbox_runtime_required` with non-retry guidance.
- Action timeout: bounded `browser_action_timeout:<action>:<seconds>s`.
- Unknown action/session mismatch: deterministic error codes.

## Verification Checklist

- [x] Sandbox provider actions execute in sandbox/container runtime.
- [x] Cross-provider session fallback is rejected.
- [x] Retention policies are enforced and logged.
- [x] Docs match actual runtime behavior and contain no host-bypass guidance.
- [x] Dedicated browser runtime uses scope-keyed container naming.
- [x] Legacy shared-executor browser runtime path removed from active behavior.
- [x] Dedicated runtime command execution uses authenticated loopback bridge with signed short-lived scope/target-bound tokens.
