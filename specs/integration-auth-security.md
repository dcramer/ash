# Integration Auth Security

> Unified secret-handling policy for tools, skills, and integration providers.

## Intent

Ash should not hand raw API keys, OAuth access tokens, refresh tokens, or client
secrets directly to tool/skill execution environments.

Authentication and authorization for sensitive external systems must be mediated by
host-managed boundaries (capabilities, provider bridges, authenticated proxies).

## Requirements

### MUST

- Tools/skills MUST NOT receive secret-like env vars by default.
- Secret-like env var names are detected by fixed built-in name patterns.
- Delivery of secret-like env vars to skills/tools is blocked by policy.
- Capability/provider auth responses MUST NOT include raw credential material.
- `credential_material` from providers is limited to opaque references (for example
  `credential_key`) and metadata.
- Provider-side credential artifacts MUST be persisted in host vault storage.
- Capability invocation responses MUST remain credential-safe.
- Identity/routing for auth and invoke MUST be token-derived (`ASH_CONTEXT_TOKEN`),
  not caller-provided ids.

### SHOULD

- External provider bridges should run with minimal inherited process environment.
- Authenticated sidecar/proxy patterns should inject authorization headers server-side
  from vault-backed references when feasible.

## Policy

The secret-delivery block is enforced as a runtime policy and is not user-configurable.
