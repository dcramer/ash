# Config

> TOML configuration loading and validation

Files: src/ash/config/loader.py, src/ash/config/models.py, src/ash/config/paths.py, src/ash/cli/app.py

## Requirements

### MUST

- Load configuration from TOML file
- Support environment variable overrides for secrets
- Validate configuration against Pydantic models
- Provide sensible defaults for all optional fields
- Support multiple LLM providers (anthropic, openai)
- Support named model configurations (`[models.<alias>]`)
- Require a `default` model alias
- Support per-skill model overrides (`[skills.<name>] model = "<alias>"`)
- Support packaged skill/provider wiring under skill config (`[skills.gog]`)
- Generate config template programmatically (no static file)

### SHOULD

- Auto-discover config file locations (~/.ash/config.toml, ./config.toml)
- Merge configs from multiple sources
- Validate provider-specific settings
- Use Haiku as default model (fast, cost-effective)
- Suggest Sonnet alias for complex tasks

### MAY

- Hot-reload configuration on file change
- Config schema export for documentation

## Interface

```python
class AshConfig(BaseModel):
    models: dict[str, ModelConfig]  # Named model configurations
    skills: dict[str, dict[str, str]]  # Per-skill config (model overrides)
    sandbox: SandboxConfig
    memory: MemoryConfig
    server: ServerConfig
    telegram: TelegramConfig | None
    brave_search: BraveSearchConfig | None
    embeddings: EmbeddingsConfig | None
    anthropic: ProviderConfig | None
    openai: ProviderConfig | None

def load_config(path: Path | None = None) -> AshConfig
def get_default_config() -> AshConfig
```

```bash
ash init [--path PATH]             # Generate config template
ash config show [--path PATH]      # Display current config
ash config validate [--path PATH]  # Validate config file
```

## Configuration

```toml
# Provider API keys
[anthropic]
api_key = "..."  # or ANTHROPIC_API_KEY env

[openai]
api_key = "..."  # or OPENAI_API_KEY env

# Named model configurations
[models.default]
provider = "openai"
model = "gpt-5.2"
temperature = 0.7
max_tokens = 4096

[models.fast]
provider = "openai"
# Suggested when available: gpt-5-mini (currently unsupported on openai-oauth)
model = "gpt-5.2"

# Per-skill model overrides
[skills.debug]
model = "codex"
allow_chat_ids = ["12345"]

[skills.defaults]
allow_chat_ids = ["12345"]

[skills.gog]
enabled = true

[skills.gog.capability_provider]
enabled = true
namespace = "gog"
command = ["gogcli", "bridge"]
timeout_seconds = 30

[skills.code-review]
model = "sonnet"

[sandbox]
timeout = 60
memory_limit = "512m"
network_mode = "bridge"
workspace_access = "rw"

[telegram]
bot_token = "..."  # or TELEGRAM_BOT_TOKEN env
allowed_users = ["123456789"]

[brave_search]
api_key = "..."  # or BRAVE_SEARCH_API_KEY env

[embeddings]
provider = "openai"
model = "text-embedding-3-small"

[server]
host = "127.0.0.1"
port = 8080
```

## Model Resolution

For skills:
1. `[skills.<name>] model` in config (per-skill override)
2. `model` in SKILL.md frontmatter
3. `"default"` fallback

Skill chat allowlist resolution:
1. `[skills.<name>].allow_chat_ids` (per-skill override, when set)
2. `[skills.defaults].allow_chat_ids` (global default)

Bundled `gog` behavior:
1. `[skills.gog].enabled = true` applies default `capabilities.providers.gog` wiring.
2. `[skills.gog.capability_provider]` can override provider settings from the skill section.
3. Explicit `[capabilities.providers.gog]` remains available for host-level overrides.

For API keys:
1. Provider config (`[anthropic].api_key`)
2. Environment variable (`ANTHROPIC_API_KEY`)

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|
| Valid TOML | AshConfig instance | Parsed and validated |
| Missing file | FileNotFoundError | No implicit default |
| Invalid TOML | TOMLDecodeError | Parse error |
| Invalid values | ValidationError | Pydantic validation |
| ENV override | Merged config | Environment fills missing |
| Missing default model | ValidationError | Required |

## Errors

| Condition | Response |
|-----------|----------|
| File not found | FileNotFoundError with search paths |
| Invalid TOML syntax | TOMLDecodeError with parse details |
| Invalid provider | ValidationError: "Invalid provider" |
| Missing required field | ValidationError with field name |
| Missing default model | ValidationError: "models.default required" |

## Verification

```bash
uv run pytest tests/test_config.py -v
ash init --path /tmp/test.toml && ash config validate --path /tmp/test.toml
```

- Generated config validates successfully
- Invalid TOML rejected
- Invalid provider rejected
- Missing default model rejected
- Environment overrides work
- Per-skill model override resolves correctly
