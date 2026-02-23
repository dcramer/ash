# Browser Subsystem

Implementation plan for adopting the service-style runtime model:
- [specs/browser-runtime-v2.md](browser-runtime-v2.md)

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

- `runtime_required` (bool): require sandbox/container runtime.
- `runtime_warmup_on_start` (bool): warm browser runtime during integration startup.
- `runtime_restart_attempts` (int): bounded restart attempts when runtime becomes unhealthy.

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

## Integration Contract

- Browser registers through integration composition hooks (`specs/subsystems.md` + `specs/integrations.md`).
- Browser RPC and tool surfaces MUST delegate to the browser subsystem manager.

## Verification Checklist

- [ ] Sandbox provider actions execute in sandbox/container runtime.
- [ ] Cross-provider session fallback is rejected.
- [ ] Retention policies are enforced and logged.
- [ ] Docs match actual runtime behavior.
