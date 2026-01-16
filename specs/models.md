# Model Selection

> Named model configurations with aliases for flexible model selection

Files: src/ash/config/models.py, src/ash/config/loader.py, src/ash/llm/registry.py, src/ash/cli/app.py

## Requirements

### MUST

- Support named model configurations via `[models.<alias>]` TOML sections
- Each named config contains: provider, model, and optionally temperature, max_tokens
- Temperature is optional (None = use provider default; omit for reasoning models that don't support it)
- Require `default` alias as the agent's primary model
- Provide `get_model(alias: str) -> ModelConfig` lookup
- API keys inherit from provider-level config if not specified per-model
- Validate alias references at config load time
- Maintain backward compatibility: `[default_llm]` maps to `models.default`
- Support CLI flag `--model <alias>` on `ash chat`

### SHOULD

- Support environment variable `ASH_MODEL` for default model override
- Provide `list_models() -> list[str]` to enumerate available aliases
- Log warning when both `[default_llm]` and `[models.default]` present

### MAY

- Support model-specific API keys via `<ALIAS>_API_KEY` env pattern
- Add `ash config models` subcommand to list aliases

## Interface

### Configuration

```toml
# Named model configurations
[models.default]
provider = "anthropic"
model = "claude-3-5-haiku"  # Fast, cheap for simple tasks
temperature = 0.7  # Optional - omit to use API default
max_tokens = 4096

[models.sonnet]
provider = "anthropic"
model = "claude-sonnet-4-5"  # More capable for complex tasks
max_tokens = 8192

[models.reasoning]
provider = "anthropic"
model = "claude-3-5-opus"
# temperature omitted for reasoning models that don't support it
max_tokens = 8192

# Provider-level API keys (shared by models using that provider)
[anthropic]
api_key = "..."  # or ANTHROPIC_API_KEY env

[openai]
api_key = "..."  # or OPENAI_API_KEY env

# Per-skill model overrides
[skills.debug]
model = "sonnet"  # Use more capable model for debugging

[skills.code-review]
model = "sonnet"  # Use more capable model for code review

# Backward compatibility (maps to models.default if no [models] section)
[default_llm]
provider = "anthropic"
model = "claude-3-5-haiku"
```

### Python Classes

```python
class ModelConfig(BaseModel):
    """Configuration for a named model."""
    provider: Literal["anthropic", "openai"]
    model: str
    temperature: float | None = None  # None = use provider default; omit for reasoning models
    max_tokens: int = 4096

class ProviderConfig(BaseModel):
    """Provider-level configuration."""
    api_key: SecretStr | None = None

class AshConfig(BaseModel):
    models: dict[str, ModelConfig] = {}
    anthropic: ProviderConfig | None = None
    openai: ProviderConfig | None = None

    def get_model(self, alias: str) -> ModelConfig:
        """Get model config by alias. Raises KeyError if not found."""
        ...

    def list_models(self) -> list[str]:
        """List available model aliases."""
        ...

    @property
    def default_model(self) -> ModelConfig:
        """Get the default model (alias 'default')."""
        ...

    def resolve_api_key(self, alias: str) -> SecretStr | None:
        """Resolve API key: provider-level > env var."""
        ...
```

### CLI

```bash
ash chat --model <alias> "prompt"   # Use specific model
ASH_MODEL=fast ash chat "prompt"    # Environment override
```

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|
| `[models.fast]` section | `get_model("fast")` returns ModelConfig | |
| `[default_llm]` without `[models]` | Migrated to `models["default"]` | Backward compatible |
| Both `[default_llm]` and `[models.default]` | `[models.default]` takes precedence | Warning logged |
| `--model fast` | Agent uses `models.fast` config | CLI override |
| `ASH_MODEL=fast` | Default model changes to "fast" | Env override |
| No API key in model, provider has key | Use provider key | Inheritance |
| `[skills.debug] model = "sonnet"` | Skill uses `models.sonnet` | Per-skill override |

## Errors

| Condition | Response |
|-----------|----------|
| Unknown alias in `--model` | ConfigError: "Unknown model alias 'X'. Available: default, fast, ..." |
| No `default` model configured | ConfigError: "No default model configured. Add [models.default] or [default_llm]" |
| Missing API key for provider | ConfigError: "No API key for provider 'anthropic'. Set ANTHROPIC_API_KEY or api_key in config" |
| Invalid provider in model | ValidationError: "Invalid provider 'X'. Must be 'anthropic' or 'openai'" |

## Verification

```bash
uv run pytest tests/test_config.py -v -k model
uv run ash chat --model fast "Hello"
ASH_MODEL=fast uv run ash chat "Hello"
```

- Config with `[models.X]` sections loads successfully
- Backward compatible `[default_llm]` still works
- `get_model()` returns correct ModelConfig
- API key inheritance works (provider > env)
- CLI `--model` flag switches model
- Invalid alias rejected with clear error
- Missing default model detected
