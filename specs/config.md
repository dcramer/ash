# Config

> TOML configuration loading and validation

Files: src/ash/config/loader.py, src/ash/config/models.py, src/ash/config/paths.py

## Requirements

### MUST

- Load configuration from TOML file
- Support environment variable overrides for secrets
- Validate configuration against Pydantic models
- Provide sensible defaults for all optional fields
- Support multiple LLM providers (anthropic, openai)
- Support multiple messaging providers (telegram)

### SHOULD

- Auto-discover config file locations (~/.ash/config.toml, ./config.toml)
- Merge configs from multiple sources
- Validate provider-specific settings

### MAY

- Hot-reload configuration on file change
- Config schema export for documentation

## Interface

```python
class AshConfig(BaseModel):
    llm: LLMConfig
    sandbox: SandboxConfig
    memory: MemoryConfig
    server: ServerConfig
    telegram: TelegramConfig | None
    brave_search: BraveSearchConfig | None

def load_config(path: Path | None = None) -> AshConfig
def get_default_config() -> AshConfig
def resolve_env_secrets(config: AshConfig) -> AshConfig
```

```bash
ash config init [--path PATH]      # Create config from template
ash config show [--path PATH]      # Display current config
ash config validate [--path PATH]  # Validate config file
```

## Configuration

```toml
[llm]
provider = "anthropic"  # or "openai"
model = "claude-sonnet-4-20250514"

[anthropic]
api_key = "..."  # or ANTHROPIC_API_KEY env

[openai]
api_key = "..."  # or OPENAI_API_KEY env

[sandbox]
timeout = 60
memory_limit = "512m"
network_mode = "none"
workspace_access = "rw"

[telegram]
bot_token = "..."  # or TELEGRAM_BOT_TOKEN env
allowed_users = ["@username"]

[brave_search]
api_key = "..."  # or BRAVE_API_KEY env

[server]
host = "0.0.0.0"
port = 8000
```

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|
| Valid TOML | AshConfig instance | Parsed and validated |
| Missing file | Error or default | Depends on context |
| Invalid TOML | TOMLDecodeError | Parse error |
| Invalid values | ValidationError | Pydantic validation |
| ENV override | Merged config | Environment takes precedence |

## Errors

| Condition | Response |
|-----------|----------|
| File not found | ConfigError: "Config file not found" |
| Invalid TOML syntax | ConfigError with parse error details |
| Invalid provider | ValidationError: "Invalid provider" |
| Missing required field | ValidationError with field name |

## Verification

```bash
uv run pytest tests/test_config.py -v
ash config validate --path config.example.toml
```

- Example config parses successfully
- Invalid TOML rejected
- Invalid provider rejected
- Environment overrides work
