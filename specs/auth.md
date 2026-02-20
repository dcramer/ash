# OAuth Authentication

> OAuth-based authentication for LLM providers that use subscription-based access instead of API keys.

Files: `src/ash/auth/oauth.py`, `src/ash/auth/storage.py`, `src/ash/llm/openai_oauth.py`, `src/ash/cli/commands/auth.py`

## Overview

Some LLM providers (like OpenAI's Codex API) allow access through OAuth rather than static API keys. This enables users with ChatGPT Plus/Pro subscriptions to use OpenAI models without a separate API key.

The auth system provides:
- PKCE OAuth login flow with local callback server + manual paste fallback
- Secure token persistence in `~/.ash/auth.json`
- Automatic token refresh before expiry
- CLI commands for credential management

## Supported Providers

### `openai-oauth`

Uses the Codex Responses API at `chatgpt.com/backend-api/codex` with ChatGPT OAuth credentials.

- **Auth endpoint**: `auth.openai.com`
- **API endpoint**: `chatgpt.com/backend-api/codex`
- **Client ID**: `app_EMoamEEZ73f0CkXaXp7hrann`
- **Callback**: `http://localhost:1455/auth/callback`
- **Scope**: `openid profile email offline_access`
- **Limitations**: Completions/streaming only — embeddings require a standard OpenAI API key.

## Configuration

```toml
# Use OpenAI OAuth as the default model (no API key needed)
[models.default]
provider = "openai-oauth"
model = "gpt-5.2"
reasoning = "medium"

# Embeddings still require a standard OpenAI API key
[openai]
api_key = "sk-..."

[embeddings]
provider = "openai"
model = "text-embedding-3-small"
```

## CLI Commands

```bash
ash auth login [openai-oauth]    # Run OAuth flow, save credentials
ash auth status                  # Show authenticated providers and token status
ash auth logout [openai-oauth]   # Remove stored credentials
```

### Login Flow

1. Generate PKCE verifier + SHA-256 challenge
2. Start HTTP callback server on `localhost:1455` (best-effort)
3. Open browser to OpenAI authorization URL
4. User authenticates in browser
5. Code received via browser redirect OR manual callback URL paste
6. Exchange code for access/refresh tokens
7. Extract `chatgpt_account_id` from JWT
8. Save credentials to `~/.ash/auth.json` (0o600 permissions)

On headless/remote machines where the browser redirect can't reach localhost, the user can paste the callback URL from their browser's address bar to complete authentication.

## Credential Storage

**File**: `~/.ash/auth.json` (permissions: `0o600`)

```json
{
  "openai-oauth": {
    "access": "eyJ...",
    "refresh": "v1.MjI...",
    "expires": 1700000000.0,
    "account_id": "acct_..."
  }
}
```

- Uses `filelock` for safe concurrent access
- Credentials are read via `AuthStorage.load(provider_id)`
- `AshConfig.resolve_oauth_credentials(provider)` is the main entry point for code needing credentials

## Token Refresh

The `OpenAIOAuthProvider` automatically refreshes tokens:

- Checks expiry before each `complete()` or `stream()` call
- Refreshes 5 minutes before expiry (`TOKEN_REFRESH_BUFFER_SECONDS = 300`)
- Updates both the in-memory client and `auth.json` on refresh
- Falls back gracefully if refresh fails (logs warning, uses existing token)

## Architecture

### Provider Creation

All LLM provider creation flows through `AshConfig.create_llm_provider_for_model(alias)`. This method handles both API key-based and OAuth-based providers, so callers don't need to know about the authentication mechanism:

```python
# In any code that needs an LLM provider:
llm = config.create_llm_provider_for_model("default")
```

This replaces the previous pattern of manually calling `resolve_api_key()` + `create_llm_provider()`, which doesn't work for OAuth providers.

### Credential Check in Chat CLI

The chat command (`ash chat`) checks credentials early before creating the agent:
- For `openai-oauth`: checks `resolve_oauth_credentials("openai-oauth")` is not None
- For other providers: checks `resolve_api_key(alias)` is not None

## Verification

```bash
# Unit tests
uv run pytest tests/test_oauth.py tests/test_auth_storage.py tests/test_openai_oauth_provider.py -v

# Integration (manual)
ash auth login openai-oauth     # Opens browser / paste callback URL
ash auth status                 # Shows "openai-oauth: valid (expires in Xh Ym)"
ash chat -m default "hello"     # Uses Codex endpoint with OAuth token
ash auth logout openai-oauth    # Removes credentials
```
