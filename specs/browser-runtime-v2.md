# Browser Runtime V2 (OpenClaw Model)

## Purpose

Adopt a service-style browser runtime model that matches the reliability characteristics of OpenClaw:

- Long-lived browser runtime inside sandbox/container
- Stable tab/target routing per session
- Fast health checks and deterministic recovery paths
- Explicit model-facing error semantics that prevent retry loops

This spec defines the architecture, contracts, migration plan, and verification requirements.

## Goals

1. Eliminate "hang-like" browser tool behavior caused by slow/failing navigation paths.
2. Make browser actions deterministic across turns and resilient to runtime drift.
3. Keep browser entirely in subsystem boundaries and integration hooks (`specs/subsystems.md`).
4. Preserve JSONL-based session/artifact state and existing browser action API surface.

## Non-Goals

1. Do not add multiple concurrent provider routing in one run (single configured provider remains).
2. Do not introduce host-side browser execution.
3. Do not expand scope to general-purpose remote browser mesh in this phase.

## Reference Model (OpenClaw Patterns to Adopt)

From `~/src/clawdbot`:

1. Dedicated browser runtime container + entrypoint
   - `Dockerfile.sandbox-browser`
   - `scripts/sandbox-browser-entrypoint.sh`
2. Runtime health and recovery decisions in service context
   - `src/browser/server-context.ts`
3. Stable target/tab identity and explicit target routing
   - `src/browser/server-context.ts` (`ensureTabAvailable`, target resolution)
4. Strong timeout normalization and per-action bounds
   - `src/browser/pw-tools-core.shared.ts`
   - `src/browser/pw-tools-core.interactions.ts`
5. Model-facing non-retry error hints
   - `src/browser/client-fetch.ts`

## Target Architecture

### 1) Browser Control Plane (inside sandbox)

Add a persistent browser control process in sandbox runtime:

- Boots Chromium once.
- Maintains CDP endpoint readiness.
- Exposes a local action API (JSON request/response).
- Keeps browser + tab/session state in-process for low-latency actions.

Implementation shape:

- New internal module (subsystem-local), e.g. `ash.browser.runtime`.
- Started/stopped by `SandboxBrowserProvider` lifecycle, not by core harness.
- Runtime process owned by provider and scoped to sandbox executor/container.

### 2) Browser Session Model

Browser session (`ash` level) maps to runtime tab/target identity:

- `session.start` creates or reserves a target.
- `session_id -> target_id` mapping persisted in browser metadata.
- On restart, provider rehydrates session-to-target mapping best-effort.
- `session.close` closes target and marks session closed in store.

No implicit shared page fallback between sessions.

### 3) Action Path

All `page.*` actions route through control plane API, not ad-hoc `python -c` scripts.

Required actions:

- `page.goto`
- `page.extract`
- `page.click`
- `page.type`
- `page.wait_for`
- `page.screenshot`

Response contract remains `BrowserActionResult` compatible.

### 4) Health + Recovery

Before action execution:

1. Fast HTTP readiness probe (`/json/version` equivalent) with short timeout.
2. Fast CDP handshake probe.
3. If unhealthy:
   - attempt runtime restart once.
   - if still unhealthy, fail with actionable error (include phase).

No unbounded wait paths.

### 5) Error Semantics (Agent UX)

Add browser errors that prevent infinite retry loops:

- Include explicit suffix:
  - "Do NOT retry browser tool; browser runtime is unavailable."
- Include operator hint:
  - "Check sandbox browser runtime/container health."

Tool errors must be short and consistent (no raw traceback dumps).

## Timeouts and Retries

### Action timeout budgets

Normalize all action timeouts with min/max clamp:

- Default action timeout: config-driven.
- Clamp: min 1s, max 120s.
- `page.goto` default lower than today for responsiveness (e.g. 15-20s).

### Navigation behavior

`page.goto` policy:

1. Try `domcontentloaded` with configured timeout.
2. On timeout/failure, fallback to `commit` with shorter bounded timeout.
3. Return best-effort current URL/title/content.

### Retry policy

- Do not retry action payloads automatically at manager layer.
- Only retry runtime connection/restart sequence (bounded, max 1 recovery cycle).

## Configuration

Keep existing public config shape, add optional internals:

`[browser.sandbox]`

- `runtime_required` (existing)
- `runtime_warmup_on_start` (default true)
- `runtime_healthcheck_interval_seconds` (future; optional)
- `runtime_restart_attempts` (default 1)

No breaking changes to required config.

## Logging and Metrics

Required logs:

1. `browser_runtime_starting`
2. `browser_runtime_ready`
3. `browser_runtime_unhealthy`
4. `browser_runtime_restarting`
5. `browser_runtime_restart_failed`
6. `browser_action_started`
7. `browser_action_succeeded`
8. `browser_action_failed`

Required fields:

- `browser.provider`
- `browser.action` (for action logs)
- `browser.session_id` when available
- `error.message` on failures

Log lines must not include empty `context=` fields.

## Integration Hooks Contract

Browser remains integration-owned:

- Setup/wiring in `BrowserIntegration`.
- Prompt guidance in structured prompt keys only.
- RPC method registration in integration hook.
- No direct feature branches in core agent entrypoints.

Reference: `specs/subsystems.md`, `specs/integrations.md`.

## Migration Plan

### Phase 1: Control Plane Foundation

1. Introduce browser runtime control process abstraction.
2. Implement health checks + restart flow.
3. Keep existing manager/tool APIs unchanged.

### Phase 2: Session/Target Routing

1. Persist `session_id -> target_id` metadata.
2. Route all actions via target identity.
3. Remove residual shared-page fallback behavior.

### Phase 3: UX/Error Hardening

1. Add non-retry model hints in browser errors.
2. Normalize timeout clamps for all actions.
3. Ensure concise actionable failure messages.

### Phase 4: Operational Hardening

1. Doctor checks validate browser runtime prerequisites for sandbox image.
2. Stats include runtime restart/unhealthy counters.
3. Add troubleshooting doc updates.

## Testing Requirements

### Unit tests

1. Runtime healthcheck/restart behavior (healthy, unhealthy, recover, fail).
2. Session-to-target mapping persistence and rehydrate.
3. Timeout clamp behavior per action.
4. Error message formatting + non-retry hint.

### Integration tests

1. Browser tool E2E: start -> goto -> extract -> screenshot.
2. Failure E2E: runtime unavailable emits non-retry guidance.
3. Recovery E2E: unhealthy runtime recovers with restart.

### Architecture tests

1. Browser wiring remains integration-owned.
2. No direct browser branches in core harness entrypoints.

## Acceptance Criteria

1. No browser action takes longer than configured timeout + bounded overhead.
2. `page.goto` failures are bounded and actionable (no opaque hangs).
3. Browser failures do not trigger model retry loops.
4. All browser tests pass; full `tests/` pass.
5. Logs clearly show runtime start, readiness, action lifecycle, and recovery attempts.

## Open Questions

1. Should runtime process be Python or a lightweight Node service in sandbox?
   - Recommendation: Python first for repo consistency and faster incremental adoption.
2. Should we keep Kernel provider in V2 scope?
   - Recommendation: keep adapter surface, but focus hardening only on sandbox provider first.
3. Should `page.goto` default timeout be reduced globally?
   - Recommendation: yes, reduce default for chat UX; allow opt-up per action.
