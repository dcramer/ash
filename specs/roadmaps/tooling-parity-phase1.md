# Tooling Parity Phase 1 Roadmap

Status date: 2026-02-24

## Scope
- WS1: Tool-output trust boundaries
- WS2: Browser depth expansion
- WS3: Vision integration expansion (replaces standalone image tool)
- WS4: Skills lifecycle hardening
- WS5: Sessions/schedule tool additions

## Progress Tracker
- WS1 Tool-output trust boundaries: In progress
- WS2 Browser depth expansion: Not started
- WS3 Vision integration expansion: Not started
- WS4 Skills lifecycle hardening: Not started
- WS5 Sessions/schedule tool additions: Not started

## WS1 Deep Dive Checklist
- [x] Add centralized sanitizer module for tool outputs.
- [x] Add trust policy and risk signal types.
- [x] Wire sanitizer into main agent tool handoff path.
- [x] Wire sanitizer into subagent/interactive executor handoff paths.
- [x] Add structured trust metadata to session-manager tool-result logs.
- [x] Add architecture guard to prevent raw `result.content` handoff.
- [x] Add sanitizer + handoff tests.
- [x] Run lint/type/tests and fix regressions.

## Notes
- Enforcement default is `warn_sanitize`.
- Config surface added under `tool_output_trust` with mode/max_chars/header options.
- Current implementation keeps original tool output in session logs while sanitizing model-visible content.

## Next Deep Dives
1. WS2 Browser depth expansion
2. WS4 Skills lifecycle hardening
3. WS3 Vision integration expansion
4. WS5 Sessions/schedule tools
