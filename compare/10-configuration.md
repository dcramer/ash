# Configuration Systems Comparison

This document compares configuration loading, validation, and management across four agent codebases: ash, archer, clawdbot, and pi-mono.

## Overview

Configuration systems handle how applications load settings at startup and runtime. Key responsibilities include:

- **Loading**: Reading configuration from files, environment variables, or other sources
- **Validation**: Ensuring configuration values are valid before use
- **Resolution**: Merging multiple sources with defined precedence
- **Secret handling**: Managing sensitive values like API keys
- **Runtime updates**: Supporting hot reload or requiring restarts

## Comparison Summary

| Feature | ash | archer | clawdbot | pi-mono |
|---------|-----|--------|----------|---------|
| **Format** | TOML | JSON | JSON5 | JSON |
| **Validation** | Pydantic | Manual | Zod | Manual |
| **Schema Export** | No | No | Yes (JSON Schema) | No |
| **Hot Reload** | No | No | Yes | No |
| **Atomic Writes** | No | No | Yes (with backup) | No |
| **Env Resolution** | `$VAR` syntax | Direct env vars | Shell env fallback | No |
| **Per-Project Settings** | No | No | No | Yes (deep merge) |
| **Search Order** | `./`, `~/.ash/`, `/etc/ash/` | `~/.archer/` only | `~/.clawdbot/` only | Global + project |
| **Secret Handling** | `SecretStr` type | Plain strings | Plain strings | Plain strings |

---

## Ash (Python)

Ash uses TOML configuration with Pydantic validation, providing strong typing and explicit secret handling.

### Configuration Path

```
src/ash/config/paths.py
src/ash/config/models.py
src/ash/config/loader.py
```

### Format and Location

TOML format with a three-tier search order:

```python
# loader.py
def _get_default_config_paths() -> list[Path]:
    return [
        Path("config.toml"),           # Current directory
        get_config_path(),             # ~/.ash/config.toml (or ASH_HOME)
        Path("/etc/ash/config.toml"),  # System-wide
    ]
```

The base directory can be overridden via `ASH_HOME` environment variable:

```python
# paths.py
@lru_cache(maxsize=1)
def get_ash_home() -> Path:
    if env_home := os.environ.get(ENV_VAR):
        return Path(env_home).expanduser().resolve()
    return Path.home() / ".ash"
```

### Validation

Pydantic models provide declarative schema validation with type coercion:

```python
# models.py
class ModelConfig(BaseModel):
    provider: Literal["anthropic", "openai"]
    model: str
    temperature: float | None = None
    max_tokens: int = 4096
    thinking: Literal["off", "minimal", "low", "medium", "high"] | None = None

class AshConfig(BaseModel):
    workspace: Path = Field(default_factory=get_workspace_path)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    anthropic: ProviderConfig | None = None
    openai: ProviderConfig | None = None
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    # ... additional sections
```

Model validators handle migrations and cross-field validation:

```python
@model_validator(mode="after")
def _migrate_default_llm(self) -> "AshConfig":
    """Migrate [default_llm] to models.default for backward compatibility."""
    if self.default_llm is None:
        return self
    if "default" in self.models:
        logger.warning(
            "Both [default_llm] and [models.default] present. "
            "Using [models.default], ignoring [default_llm]."
        )
        return self
    # ... migration logic
```

### Secret Handling

Pydantic's `SecretStr` type prevents accidental secret exposure in logs:

```python
class ProviderConfig(BaseModel):
    api_key: SecretStr | None = None

class TelegramConfig(BaseModel):
    bot_token: SecretStr | None = None
```

Environment variables provide fallback for secrets not in config:

```python
def _resolve_env_secrets(config: dict[str, Any]) -> dict[str, Any]:
    provider_env_vars = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    for provider, env_var in provider_env_vars.items():
        if provider in config:
            _set_secret_from_env(config[provider], "api_key", env_var)
    # ...
```

### Environment Variable Resolution

The `[env]` section supports `$VAR` syntax for referencing environment variables:

```python
def get_resolved_env(self) -> dict[str, str]:
    resolved = {}
    for name, value in self.env.items():
        if value.startswith("$"):
            env_var = value[1:]
            resolved[name] = os.environ.get(env_var, "")
        else:
            resolved[name] = value
    return resolved
```

### Strengths

- Strong typing with Pydantic catches errors at load time
- `SecretStr` prevents accidental secret logging
- Multi-tier search order supports local, user, and system configs
- Migration support for deprecated configuration formats

### Limitations

- No hot reload - requires restart for config changes
- No schema export for external tooling
- No per-project configuration support

---

## Archer (TypeScript)

Archer uses the simplest configuration approach: direct JSON files with manual parsing and environment variable overrides.

### Configuration Path

```
src/main.ts (inline)
```

### Format and Location

JSON files in `~/.archer/`:

```typescript
// main.ts
interface TelegramConfig {
  token?: string;
  permittedUsers?: string[];
}

function getTelegramConfig(): TelegramConfig {
  const config: TelegramConfig = {};

  // Environment variable takes precedence
  if (process.env.ARCHER_TELEGRAM_TOKEN) {
    config.token = process.env.ARCHER_TELEGRAM_TOKEN;
  }

  // Then check JSON file
  const configPath = join(homedir(), ".archer", "telegram.json");
  if (existsSync(configPath)) {
    try {
      const data = JSON.parse(readFileSync(configPath, "utf-8"));
      if (!config.token && data.token) {
        config.token = data.token;
      }
      if (Array.isArray(data.permittedUsers)) {
        config.permittedUsers = data.permittedUsers.map(String);
      }
    } catch {
      // Ignore parse errors
    }
  }

  return config;
}
```

### Validation

Manual validation with runtime error messages:

```typescript
if (!ARCHER_TELEGRAM_TOKEN) {
  console.error("Missing Telegram token.");
  console.error("");
  console.error(
    "Either set ARCHER_TELEGRAM_TOKEN environment variable, or create ~/.archer/telegram.json:"
  );
  console.error('  {"token": "your-bot-token-from-botfather"}');
  process.exit(1);
}
```

### Strengths

- Simple and easy to understand
- Environment variables take clear precedence
- Minimal dependencies

### Limitations

- No schema validation - invalid config values discovered at runtime
- Manual error handling for each field
- No type safety for config structure
- Config structure embedded in main entry point

---

## Clawdbot (TypeScript)

Clawdbot has the most sophisticated configuration system: JSON5 format, Zod validation with JSON Schema export, hot reload, atomic writes with backup, and shell environment fallback.

### Configuration Paths

```
src/config/paths.ts
src/config/io.ts
src/config/zod-schema.ts
src/config/types.ts
```

### Format and Location

JSON5 format (allows comments, trailing commas) in `~/.clawdbot/clawdbot.json`:

```typescript
// paths.ts
export function resolveConfigPath(
  env: NodeJS.ProcessEnv = process.env,
  stateDir: string = resolveStateDir(env, os.homedir)
): string {
  const override = env.CLAWDBOT_CONFIG_PATH?.trim();
  if (override) return override;
  return path.join(stateDir, "clawdbot.json");
}

export function resolveStateDir(
  env: NodeJS.ProcessEnv = process.env,
  homedir: () => string = os.homedir
): string {
  const override =
    env.CLAWDBOT_STATE_DIR?.trim() || env.CLAWDIS_STATE_DIR?.trim();
  if (override) return resolveUserPath(override);
  return path.join(homedir(), ".clawdbot");
}
```

### Validation

Comprehensive Zod schema with deeply nested validation:

```typescript
// zod-schema.ts
const ModelDefinitionSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  api: ModelApiSchema.optional(),
  reasoning: z.boolean(),
  input: z.array(z.union([z.literal("text"), z.literal("image")])),
  cost: z.object({
    input: z.number(),
    output: z.number(),
    cacheRead: z.number(),
    cacheWrite: z.number(),
  }),
  contextWindow: z.number().positive(),
  maxTokens: z.number().positive(),
  headers: z.record(z.string(), z.string()).optional(),
  compat: ModelCompatSchema,
});

export const ClawdbotSchema = z.object({
  env: z.object({
    shellEnv: z.object({
      enabled: z.boolean().optional(),
      timeoutMs: z.number().int().nonnegative().optional(),
    }).optional(),
    vars: z.record(z.string(), z.string()).optional(),
  }).catchall(z.string()).optional(),
  // ... extensive schema with 1500+ lines
});
```

Custom refinements provide cross-field validation:

```typescript
const TelegramAccountSchema = TelegramAccountSchemaBase.superRefine(
  (value, ctx) => {
    requireOpenAllowFrom({
      policy: value.dmPolicy,
      allowFrom: value.allowFrom,
      ctx,
      path: ["allowFrom"],
      message:
        'telegram.dmPolicy="open" requires telegram.allowFrom to include "*"',
    });
  }
);
```

### Atomic Writes with Backup

Configuration writes use atomic rename with automatic backup:

```typescript
// io.ts
async function writeConfigFile(cfg: ClawdbotConfig) {
  const dir = path.dirname(configPath);
  await deps.fs.promises.mkdir(dir, { recursive: true, mode: 0o700 });
  const json = JSON.stringify(applyModelDefaults(cfg), null, 2)
    .trimEnd()
    .concat("\n");

  // Write to temp file first
  const tmp = path.join(
    dir,
    `${path.basename(configPath)}.${process.pid}.${crypto.randomUUID()}.tmp`
  );

  await deps.fs.promises.writeFile(tmp, json, {
    encoding: "utf-8",
    mode: 0o600,
  });

  // Create backup
  await deps.fs.promises
    .copyFile(configPath, `${configPath}.bak`)
    .catch(() => {});

  // Atomic rename
  try {
    await deps.fs.promises.rename(tmp, configPath);
  } catch (err) {
    // Handle Windows EPERM/EEXIST edge case
    const code = (err as { code?: string }).code;
    if (code === "EPERM" || code === "EEXIST") {
      await deps.fs.promises.copyFile(tmp, configPath);
      await deps.fs.promises.chmod(configPath, 0o600).catch(() => {});
      await deps.fs.promises.unlink(tmp).catch(() => {});
      return;
    }
    throw err;
  }
}
```

### Shell Environment Fallback

Clawdbot can load environment variables from shell initialization:

```typescript
const SHELL_ENV_EXPECTED_KEYS = [
  "OPENAI_API_KEY",
  "ANTHROPIC_API_KEY",
  "ANTHROPIC_OAUTH_TOKEN",
  "GEMINI_API_KEY",
  // ... many more
];

// In loadConfig():
const enabled =
  shouldEnableShellEnvFallback(deps.env) ||
  cfg.env?.shellEnv?.enabled === true;
if (enabled) {
  loadShellEnvFallback({
    enabled: true,
    env: deps.env,
    expectedKeys: SHELL_ENV_EXPECTED_KEYS,
    logger: deps.logger,
    timeoutMs: cfg.env?.shellEnv?.timeoutMs ??
      resolveShellEnvFallbackTimeoutMs(deps.env),
  });
}
```

### Config Snapshot for UI

The system provides structured snapshots for config editor UI:

```typescript
export type ConfigFileSnapshot = {
  path: string;
  exists: boolean;
  raw: string | null;
  parsed: unknown;
  valid: boolean;
  config: ClawdbotConfig;
  issues: Array<{ path: string; message: string }>;
  legacyIssues: LegacyConfigIssue[];
};

async function readConfigFileSnapshot(): Promise<ConfigFileSnapshot> {
  // Returns structured info about config state for UI display
}
```

### Strengths

- JSON5 allows comments and trailing commas
- Comprehensive Zod validation with detailed error messages
- Atomic writes prevent corruption
- Automatic backup before writes
- Shell environment fallback for headless environments
- Config snapshot API for UI editors

### Limitations

- No per-project settings
- Complex schema (1500+ lines) is hard to maintain
- Single config location (no search order)

---

## Pi-Mono (TypeScript)

Pi-mono implements global + per-project settings with deep merge strategy, suitable for developer tools that need project-specific configuration.

### Configuration Path

```
packages/coding-agent/src/core/settings-manager.ts
```

### Format and Location

JSON files with two locations:

```typescript
// settings-manager.ts
static create(cwd: string = process.cwd(), agentDir: string = getAgentDir()): SettingsManager {
  const settingsPath = join(agentDir, "settings.json");           // Global
  const projectSettingsPath = join(cwd, CONFIG_DIR_NAME, "settings.json"); // Per-project
  const globalSettings = SettingsManager.loadFromFile(settingsPath);
  return new SettingsManager(settingsPath, projectSettingsPath, globalSettings, true);
}
```

### Deep Merge Strategy

Project settings override global settings with recursive object merging:

```typescript
function deepMergeSettings(base: Settings, overrides: Settings): Settings {
  const result: Settings = { ...base };

  for (const key of Object.keys(overrides) as (keyof Settings)[]) {
    const overrideValue = overrides[key];
    const baseValue = base[key];

    if (overrideValue === undefined) {
      continue;
    }

    // For nested objects, merge recursively
    if (
      typeof overrideValue === "object" &&
      overrideValue !== null &&
      !Array.isArray(overrideValue) &&
      typeof baseValue === "object" &&
      baseValue !== null &&
      !Array.isArray(baseValue)
    ) {
      (result as Record<string, unknown>)[key] = {
        ...baseValue,
        ...overrideValue,
      };
    } else {
      // For primitives and arrays, override value wins
      (result as Record<string, unknown>)[key] = overrideValue;
    }
  }

  return result;
}
```

### Settings Interface

TypeScript interface defines the configuration shape:

```typescript
export interface Settings {
  lastChangelogVersion?: string;
  defaultProvider?: string;
  defaultModel?: string;
  defaultThinkingLevel?: "off" | "minimal" | "low" | "medium" | "high" | "xhigh";
  steeringMode?: "all" | "one-at-a-time";
  theme?: string;
  compaction?: CompactionSettings;
  retry?: RetrySettings;
  skills?: SkillsSettings;
  terminal?: TerminalSettings;
  images?: ImageSettings;
  thinkingBudgets?: ThinkingBudgetsSettings;
}
```

### Migration Support

Settings are migrated on load for backward compatibility:

```typescript
private static migrateSettings(settings: Record<string, unknown>): Settings {
  // Migrate queueMode -> steeringMode
  if ("queueMode" in settings && !("steeringMode" in settings)) {
    settings.steeringMode = settings.queueMode;
    delete settings.queueMode;
  }
  return settings as Settings;
}
```

### In-Memory Mode for Testing

Factory method creates non-persistent settings:

```typescript
static inMemory(settings: Partial<Settings> = {}): SettingsManager {
  return new SettingsManager(null, null, settings, false);
}
```

### Save-on-Change Pattern

Settings are written immediately when changed:

```typescript
setDefaultProvider(provider: string): void {
  this.globalSettings.defaultProvider = provider;
  this.save();
}

private save(): void {
  if (this.persist && this.settingsPath) {
    try {
      // Re-read current file to preserve externally-added settings
      const currentFileSettings = SettingsManager.loadFromFile(this.settingsPath);
      const mergedSettings = deepMergeSettings(currentFileSettings, this.globalSettings);
      this.globalSettings = mergedSettings;
      writeFileSync(this.settingsPath, JSON.stringify(this.globalSettings, null, 2), "utf-8");
    } catch (error) {
      console.error(`Warning: Could not save settings file: ${error}`);
    }
  }
  // Re-merge to update active settings
  const projectSettings = this.loadProjectSettings();
  this.settings = deepMergeSettings(this.globalSettings, projectSettings);
}
```

### Strengths

- Per-project settings support
- Deep merge preserves nested defaults
- Migration support for schema evolution
- In-memory mode for testing
- Re-reads file before save to preserve external edits

### Limitations

- No schema validation - invalid values cause runtime errors
- No environment variable support
- Manual getter/setter pattern is verbose
- Single-level deep merge (not recursive beyond one level)

---

## Key Differences

### Validation Approach

| Codebase | Approach | Error Discovery |
|----------|----------|-----------------|
| ash | Pydantic declarative | Load time |
| archer | Manual checks | Runtime |
| clawdbot | Zod declarative | Load time |
| pi-mono | TypeScript interface | Compile time (partial) |

### Configuration Hierarchy

| Codebase | Layers | Merge Strategy |
|----------|--------|----------------|
| ash | `./` > `~/` > `/etc/` | First found wins |
| archer | `~/` only | Single source |
| clawdbot | `~/` only | Single source |
| pi-mono | Global + Project | Deep merge (project wins) |

### Environment Variable Integration

| Codebase | Pattern |
|----------|---------|
| ash | `$VAR` syntax in `[env]`, fallback for secrets |
| archer | Direct `process.env` with precedence over file |
| clawdbot | Shell env fallback, `env.vars` section |
| pi-mono | None |

### Secret Handling

| Codebase | Approach |
|----------|----------|
| ash | `SecretStr` type prevents logging |
| archer | Plain strings |
| clawdbot | Plain strings |
| pi-mono | N/A (no secrets in settings) |

---

## Recommendations

### For New Projects

1. **Use declarative validation** (Pydantic or Zod) - catches errors at load time rather than runtime
2. **Support environment variable fallback** - essential for containerized deployments
3. **Consider per-project settings** if building developer tools
4. **Use atomic writes** if config can be modified at runtime
5. **Handle secrets explicitly** - use dedicated types or separate files

### Potential Improvements

**ash:**
- Add JSON Schema export from Pydantic for editor integration
- Consider per-project overrides for workspace-specific settings

**archer:**
- Extract config to dedicated module
- Add Zod or similar validation

**clawdbot:**
- Consider splitting massive schema into smaller modules
- Add per-project settings for workspace customization

**pi-mono:**
- Add validation layer (Zod)
- Support environment variable overrides for CI/CD

### Common Patterns Worth Adopting

1. **clawdbot's atomic write** pattern prevents corruption during power loss
2. **ash's secret handling** with `SecretStr` should be adopted by TypeScript projects
3. **pi-mono's deep merge** is essential for developer tools
4. **clawdbot's config snapshot** enables sophisticated config editor UIs
