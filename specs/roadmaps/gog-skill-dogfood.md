# GOG Skill Dogfood Roadmap

Status date: 2026-02-24

## Goal

Ship `gog` as a bundled capability-provider + skill workflow for first-party
dogfooding.

## Architecture Contract

- Skill stays in the skill layer (`SKILL.md`) and uses `ash-sb capability`.
- Provider/container runtime stays in host config (`config.toml`), not in skill metadata.
- Identity, chat scope, and effective user are always derived from signed
  `ASH_CONTEXT_TOKEN`/`context_token`.
- Capability IDs stay globally namespaced (`gog.email`, `gog.calendar`).

## Current Status

- [x] Capability manager + RPC + sandbox CLI contract in place.
- [x] Bridge envelope contract (`bridge-v1`) with strict validation in place.
- [x] Signed context token propagation to bridge providers in place.
- [x] Skill capability preflight + DM/sensitive access policy in place.
- [x] Bundled opt-in `gog` skill scaffolded (`src/ash/skills/bundled/gog/SKILL.md`).
- [x] Bundled `gogcli` bridge runtime shipped with Ash
  (`src/ash/skills/bundled/gog/gogcli_bridge.py`, `gogcli` entrypoint).
- [ ] Provider persistence for credential/account state finalized.
- [ ] End-to-end Google OAuth and operation tests completed.

## Enablement Example

```toml
# config.toml

[skills.gog]
enabled = true

[skills.gog.capability_provider]
enabled = true
namespace = "gog"
command = ["gogcli", "bridge"]
timeout_seconds = 30

[skills.defaults]
allow_chat_ids = ["<dm-chat-id>"]  # optional global guardrail
```

## Remaining Work Plan

1. Bridge runtime
- Accept `bridge-v1` envelopes on stdin and return envelopes on stdout.
- Implement methods: `definitions`, `auth_begin`, `auth_complete`, `invoke`.
- Verify `context_token` with `ASH_CONTEXT_TOKEN_SECRET` before reading caller scope.

2. Capability surface
- Register `gog.email` and `gog.calendar`.
- Keep operation names stable and namespaced by capability.
- Return credential-safe outputs only.

3. Auth and storage
- Store credentials keyed by `(effective_user_id, capability_id, account_ref)`.
- Keep operation/artifact data isolated by `(effective_user_id, capability_id)`.
- Ensure auth flow handles are short-lived and user-bound.

4. Security and reliability
- Fail closed on missing/invalid token verification.
- Never trust caller-provided `user_id`/`chat_id` fields.
- Never return raw OAuth artifacts.

## Open Design Questions

- Whether auth flows should be capability-specific (`gog.email`) or integration-wide
  under a single `gog` auth handle.
- Whether to introduce first-class integration namespaces beyond capability IDs for
  future multi-provider auth orchestration.

## Core Changes Needed?

For the bundled-shipping model (skill + provider runtime in Ash), no additional
core architecture changes are required.

Optional future improvements:

- Add installer UX to scaffold `[skills.gog]` plus
  `[skills.gog.capability_provider]` defaults.
- Add provider health diagnostics in `ash doctor` for capability bridges.
