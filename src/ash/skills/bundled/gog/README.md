# Bundled `gog` Skill

This directory contains Ash's bundled dogfood implementation for Gmail/Calendar
capability workflows.

## What Is Here

- `SKILL.md`: the skill instruction surface used by `use_skill`.
- `scripts/gogcli_bridge.py`: bridge runtime implementing the capability provider
  subprocess contract (`bridge-v1`).

The bridge is exposed through the packaged CLI entrypoint:

```bash
gogcli bridge
```

## Security Model

`gog` is intentionally split into two parts:

- Skill layer (`SKILL.md`): untrusted prompt/instruction surface.
- Host/provider layer (`gogcli bridge`): trusted runtime boundary for auth and
  operations.

Security invariants:

- Skill operations go through `ash-sb capability ...` (not direct credentials).
- Caller identity/scope comes from signed `context_token` verification, not
  caller-provided `user_id`/`chat_id`.
- Sensitive capabilities are DM-only by policy (`private` chat type).
- Account/credential state is user-scoped by verified identity.
- Provider responses must not include raw OAuth artifacts.

## Enablement

```toml
[skills.gog]
enabled = true

[skills.gog.capability_provider]
enabled = true
namespace = "gog"
command = ["gogcli", "bridge"]
timeout_seconds = 30
```

`[skills.gog]` enables the bundled skill.
`[skills.gog.capability_provider]` wires the bundled bridge command.

## Notes

- This is bundled for first-party dogfooding.
- The contract remains the same as installed/workspace skills and capability
  providers.
