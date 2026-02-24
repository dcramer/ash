# Bundled Skills

Bundled skills are shipped with Ash for dogfooding and baseline workflows.
They are still ordinary skills, not privileged plugins.

## Contract

- Same `SKILL.md` schema as workspace/installed skills.
- Same validation and parsing rules.
- Same runtime execution path (`use_skill` subagent execution).
- Same security model and policy checks (`sensitive`, chat policy, capability preflight).
- No direct integration-hook wiring from the skill itself.

## Security Boundary

Bundled skills are untrusted instruction surfaces from a runtime perspective.
Privileged behavior must still go through trusted host interfaces:

- `ash-sb` commands
- RPC methods
- capability APIs

For sensitive integrations (email/calendar/etc.):

- Skills declare required capability IDs (for example `gog.email`, `gog.calendar`).
- Host config provides provider runtime wiring in `config.toml`:

```toml
[capabilities.providers.gog]
enabled = true
namespace = "gog"
command = ["gogcli", "bridge"]
timeout_seconds = 30
```

Skill metadata must not define provider/container command wiring.

## `gog` Packaging Model

`gog` is intentionally a two-part integration:

1. Skill surface: `SKILL.md` instructions and capability requirements.
2. Provider surface: external `gogcli bridge` runtime implementing capability contract.

This can live in a single third-party repo/package as long as Ash wiring stays split:

- preset toggle: `[bundles.gog] enabled = true`
- optional explicit overrides:
  - skill enablement: `[skills.gog] enabled = true`
  - provider wiring: `[capabilities.providers.gog] ...`

## Layout

Bundled skills live under this directory:

```text
src/ash/skills/bundled/<skill-name>/SKILL.md
```

Keep bundled skills compatible with external distribution so they can be moved to
or mirrored from third-party skill sources without semantic changes.
