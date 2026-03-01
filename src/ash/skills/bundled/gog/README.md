# Bundled `google` Bridge

This directory contains the capability provider bridge for Gmail/Calendar
workflows. The skill instructions live in `src/ash/integrations/skills/capabilities/google/SKILL.md`.

## What Is Here

- `scripts/gogcli_bridge.py`: bridge runtime implementing the capability provider
  subprocess contract (`bridge-v1`).

The bridge is exposed through the packaged CLI entrypoint:

```bash
gogcli bridge
```

## Getting Started

Add this to `config.toml`:

```toml
[skills.google]
enabled = true
```

That's it. The capability provider bridge is auto-wired when the skill is enabled.

Start a private chat (`ash chat`) and ask the agent about your email or calendar —
it will check auth status and walk you through Google OAuth setup on first use.

Optional chat allowlist guardrails:

```toml
[skills.defaults]
allow_chat_ids = ["<dm-chat-id>"]

# or per-skill override:
[skills.google]
enabled = true
allow_chat_ids = ["<dm-chat-id>"]
```

## Security Model

The google skill is intentionally split into two parts:

- Skill layer (`SKILL.md`): untrusted prompt/instruction surface.
- Host/provider layer (`gogcli bridge`): trusted runtime boundary for auth and
  operations.

Security invariants:

- Skill operations go through `ash-sb capability ...` (not direct credentials).
- Caller identity/scope comes from signed `context_token` verification, not
  caller-provided `user_id`/`chat_id`.
- Sensitive capabilities are DM-only by policy (`private` chat type).
- Account/credential state is user-scoped by verified identity.
- OAuth exchange artifacts and credential material are stored in host vault
  records (state keeps only vault references).
- Provider responses must not include raw OAuth artifacts.

## Runtime Paths

By default, the bridge stores:

- provider state at `~/.ash/gogcli/state.json`
- credential artifacts in the host vault at `~/.ash/vault`

For isolated local testing, override before launching Ash:

```bash
export GOGCLI_STATE_PATH=/tmp/ash-gog/state.json
export GOGCLI_VAULT_PATH=/tmp/ash-gog/vault
```

## Testing

Contract/integration tests (no external Google dependency):

```bash
uv run pytest tests/test_gogcli_bridge.py tests/test_gog_capability_e2e.py
```

Interactive local check:

1. Enable `[skills.google]` in `config.toml`.
2. Start chat (`ash chat`) in a private context.
3. Ask the agent to check your email — it will verify capabilities, handle auth, and invoke.

Current behavior note: the bundled bridge is a dogfood/reference provider. It
validates auth/isolation contracts and returns deterministic sample outputs for
email/calendar operations.

## Notes

- This is bundled for first-party dogfooding.
- The contract remains the same as installed/workspace skills and capability
  providers.
