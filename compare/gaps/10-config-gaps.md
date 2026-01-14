# Configuration System Gap Analysis

This document analyzes gaps between Ash's configuration system and the reference implementations in Clawdbot and pi-mono.

**Files Analyzed:**
- Ash: `/home/dcramer/src/ash/src/ash/config/models.py`
- Ash: `/home/dcramer/src/ash/src/ash/config/loader.py`
- Ash: `/home/dcramer/src/ash/src/ash/cli/commands/config.py`
- Clawdbot: `/home/dcramer/src/clawdbot/src/config/io.ts`
- Clawdbot: `/home/dcramer/src/clawdbot/src/config/zod-schema.ts`
- Pi-mono: `/home/dcramer/src/pi-mono/packages/coding-agent/src/core/settings-manager.ts`

---

## Gap 1: Hot Config Reload

### What Ash is Missing

Ash loads configuration once at startup via `load_config()` and requires a full process restart to pick up changes:

```python
# loader.py lines 73-112
def load_config(path: Path | None = None) -> AshConfig:
    """Load configuration from TOML file.
    ...
    """
    config_path: Path | None = None
    # ... find and load config once ...
    with config_path.open("rb") as f:
        raw_config = tomllib.load(f)
    return AshConfig.model_validate(raw_config)
```

There is no mechanism to watch for file changes or reload configuration without restarting the server.

### Reference Implementation (Clawdbot)

Clawdbot supports hot config reload with a hybrid mode that can reload configuration without restarting:

```typescript
// io.ts lines 131-205 - createConfigIO factory
export function createConfigIO(overrides: ConfigIoDeps = {}) {
  const deps = normalizeDeps(overrides);
  const configPath = resolveConfigPathForDeps(deps);

  function loadConfig(): ClawdbotConfig {
    // Re-reads config file on every call
    if (!deps.fs.existsSync(configPath)) {
      // ...
      return {};
    }
    const raw = deps.fs.readFileSync(configPath, "utf-8");
    const parsed = deps.json5.parse(raw);
    // ... validate and return
  }

  return {
    configPath,
    loadConfig,  // Can be called repeatedly
    readConfigFileSnapshot,
    writeConfigFile,
  };
}
```

The gateway configuration (zod-schema.ts lines 1467-1480) shows the reload modes:

```typescript
// zod-schema.ts lines 1467-1480
reload: z
  .object({
    mode: z
      .union([
        z.literal("off"),
        z.literal("restart"),
        z.literal("hot"),
        z.literal("hybrid"),
      ])
      .optional(),
    debounceMs: z.number().int().min(0).optional(),
  })
  .optional(),
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/config/loader.py`
- `/home/dcramer/src/ash/src/ash/config/models.py`
- `/home/dcramer/src/ash/src/ash/config/__init__.py`
- `/home/dcramer/src/ash/src/ash/core/services.py` (new file or existing service layer)

### Concrete Python Code Changes

```python
# config/loader.py - Add new classes and functions

import asyncio
import logging
from pathlib import Path
from typing import Callable, Literal
from watchfiles import awatch, Change

logger = logging.getLogger(__name__)

# New types
ConfigChangeCallback = Callable[["AshConfig"], None]
ReloadMode = Literal["off", "restart", "hot", "hybrid"]


class ConfigWatcher:
    """Watches config file for changes and triggers reload.

    Modes:
    - off: No watching, config is static
    - restart: Signal that restart is needed (don't auto-reload)
    - hot: Reload config in-place without restart
    - hybrid: Hot reload for safe settings, signal restart for others
    """

    def __init__(
        self,
        config_path: Path,
        mode: ReloadMode = "off",
        debounce_ms: int = 500,
    ) -> None:
        self._config_path = config_path
        self._mode = mode
        self._debounce_ms = debounce_ms
        self._callbacks: list[ConfigChangeCallback] = []
        self._watch_task: asyncio.Task[None] | None = None
        self._current_config: AshConfig | None = None

    def add_callback(self, callback: ConfigChangeCallback) -> None:
        """Register a callback for config changes."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: ConfigChangeCallback) -> None:
        """Unregister a callback."""
        self._callbacks.remove(callback)

    @property
    def current_config(self) -> AshConfig | None:
        """Get the currently loaded config."""
        return self._current_config

    def load(self) -> AshConfig:
        """Load config from file (initial or reload)."""
        self._current_config = load_config(self._config_path)
        return self._current_config

    async def start(self) -> None:
        """Start watching for config changes."""
        if self._mode == "off":
            return

        if self._watch_task is not None:
            return

        self._watch_task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        """Stop watching for config changes."""
        if self._watch_task is None:
            return

        self._watch_task.cancel()
        try:
            await self._watch_task
        except asyncio.CancelledError:
            pass
        self._watch_task = None

    async def _watch_loop(self) -> None:
        """Watch config file for changes."""
        try:
            async for changes in awatch(
                self._config_path.parent,
                debounce=self._debounce_ms,
            ):
                for change_type, path in changes:
                    if Path(path) != self._config_path:
                        continue
                    if change_type in (Change.modified, Change.added):
                        await self._handle_change()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Config watcher error: {e}")

    async def _handle_change(self) -> None:
        """Handle config file change based on mode."""
        logger.info(f"Config file changed, mode={self._mode}")

        if self._mode == "restart":
            logger.warning("Config changed - restart required to apply")
            return

        try:
            new_config = load_config(self._config_path)
        except Exception as e:
            logger.error(f"Failed to reload config: {e}")
            return

        if self._mode == "hybrid":
            # Check if changes require restart
            if self._requires_restart(self._current_config, new_config):
                logger.warning("Config changed - restart required for some settings")
                # Still apply safe changes
                new_config = self._merge_safe_changes(
                    self._current_config, new_config
                )

        self._current_config = new_config

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(new_config)
            except Exception as e:
                logger.error(f"Config change callback error: {e}")

    def _requires_restart(
        self, old: AshConfig | None, new: AshConfig
    ) -> bool:
        """Check if changes require restart (non-hot-reloadable settings)."""
        if old is None:
            return False

        # Settings that require restart
        restart_required = [
            old.server != new.server,  # Server host/port
            old.sandbox.image != new.sandbox.image,  # Sandbox image
            old.memory.database_path != new.memory.database_path,  # DB path
        ]
        return any(restart_required)

    def _merge_safe_changes(
        self, old: AshConfig | None, new: AshConfig
    ) -> AshConfig:
        """Merge only hot-reloadable settings from new config."""
        if old is None:
            return new

        # Create copy with safe changes applied
        # Keep restart-required settings from old config
        return AshConfig(
            workspace=new.workspace,  # Safe to change
            models=new.models,  # Safe to change
            anthropic=new.anthropic,  # Safe to change
            openai=new.openai,  # Safe to change
            telegram=new.telegram,  # Safe to change (webhook will reconnect)
            sandbox=old.sandbox,  # Keep old - requires restart
            server=old.server,  # Keep old - requires restart
            memory=new.memory.model_copy(
                update={"database_path": old.memory.database_path}
            ),
            conversation=new.conversation,  # Safe to change
            embeddings=new.embeddings,  # Safe to change
            brave_search=new.brave_search,  # Safe to change
            sentry=new.sentry,  # Safe to change
            env=new.env,  # Safe to change
            agents=new.agents,  # Safe to change
        )


# Add to config models.py - reload configuration
class ReloadConfig(BaseModel):
    """Configuration for hot reload behavior."""

    mode: Literal["off", "restart", "hot", "hybrid"] = "off"
    debounce_ms: int = Field(default=500, ge=0)


# Update AshConfig to include reload settings
class AshConfig(BaseModel):
    # ... existing fields ...
    reload: ReloadConfig = Field(default_factory=ReloadConfig)
```

### Estimated Effort

**Medium** - 3-4 hours

- Add `watchfiles` dependency
- Implement `ConfigWatcher` class
- Integrate with server startup
- Add reload config to schema
- Test hot reload behavior

### Priority

**Medium** - Useful for development and long-running server deployments where config tweaks are common.

---

## Gap 2: Per-Project Settings

### What Ash is Missing

Ash has only global configuration at `~/.ash/config.toml`. There is no way to override settings per-project or per-workspace:

```python
# loader.py lines 14-20
def _get_default_config_paths() -> list[Path]:
    """Get ordered list of default config file locations."""
    return [
        Path("config.toml"),  # Current directory - but this is full config, not override
        get_config_path(),  # ~/.ash/config.toml
        Path("/etc/ash/config.toml"),  # System-wide
    ]
```

While Ash supports a `config.toml` in the current directory, this is a full replacement, not a layered override.

### Reference Implementation (Pi-mono)

Pi-mono has a `SettingsManager` that merges global and project-level settings with deep merging:

```typescript
// settings-manager.ts lines 104-131
export class SettingsManager {
  private settingsPath: string | null;
  private projectSettingsPath: string | null;
  private globalSettings: Settings;
  private settings: Settings;

  private constructor(
    settingsPath: string | null,
    projectSettingsPath: string | null,
    initialSettings: Settings,
    persist: boolean,
  ) {
    this.settingsPath = settingsPath;
    this.projectSettingsPath = projectSettingsPath;
    this.persist = persist;
    this.globalSettings = initialSettings;
    const projectSettings = this.loadProjectSettings();
    // Deep merge: project settings override global
    this.settings = deepMergeSettings(this.globalSettings, projectSettings);
  }

  /** Create a SettingsManager that loads from files */
  static create(cwd: string = process.cwd(), agentDir: string = getAgentDir()): SettingsManager {
    const settingsPath = join(agentDir, "settings.json");
    const projectSettingsPath = join(cwd, CONFIG_DIR_NAME, "settings.json");
    const globalSettings = SettingsManager.loadFromFile(settingsPath);
    return new SettingsManager(settingsPath, projectSettingsPath, globalSettings, true);
  }
}
```

The deep merge function (lines 74-102) recursively merges nested objects:

```typescript
// settings-manager.ts lines 74-102
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
      (result as Record<string, unknown>)[key] = { ...baseValue, ...overrideValue };
    } else {
      // For primitives and arrays, override value wins
      (result as Record<string, unknown>)[key] = overrideValue;
    }
  }

  return result;
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/config/loader.py`
- `/home/dcramer/src/ash/src/ash/config/models.py`
- `/home/dcramer/src/ash/src/ash/config/paths.py`

### Concrete Python Code Changes

```python
# config/loader.py - Add project settings support

from typing import Any
from pydantic import BaseModel


def deep_merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries.

    - Override values take precedence
    - Nested dicts are merged recursively
    - Lists and primitives from override replace base entirely
    """
    result = base.copy()

    for key, override_value in overrides.items():
        if override_value is None:
            continue

        base_value = result.get(key)

        # Recursive merge for nested dicts (not lists)
        if (
            isinstance(override_value, dict)
            and isinstance(base_value, dict)
        ):
            result[key] = deep_merge_dict(base_value, override_value)
        else:
            # Primitives and lists: override wins
            result[key] = override_value

    return result


def _get_project_config_path(workspace: Path | None = None) -> Path | None:
    """Get project-specific config file path.

    Looks for .ash/config.toml in the workspace directory.
    """
    if workspace is None:
        workspace = Path.cwd()

    project_config = workspace / ".ash" / "config.toml"
    if project_config.exists():
        return project_config
    return None


def load_config(
    path: Path | None = None,
    workspace: Path | None = None,
    merge_project: bool = True,
) -> AshConfig:
    """Load configuration from TOML file with optional project overlay.

    Args:
        path: Explicit path to global config file.
        workspace: Workspace directory for project-level config.
        merge_project: Whether to merge .ash/config.toml from workspace.

    Config resolution order (later overrides earlier):
    1. System config (/etc/ash/config.toml)
    2. User config (~/.ash/config.toml)
    3. Explicit path (if provided)
    4. Project config (.ash/config.toml in workspace)

    Returns:
        Validated AshConfig instance.

    Raises:
        FileNotFoundError: If no config file is found.
        ValueError: If config file is invalid.
    """
    # Load base config (existing logic)
    config_path = _find_config_path(path)
    if config_path is None:
        raise FileNotFoundError(
            f"No config file found. Searched: {', '.join(str(p) for p in _get_default_config_paths())}"
        )

    with config_path.open("rb") as f:
        raw_config = tomllib.load(f)

    # Merge project config if enabled
    if merge_project:
        project_path = _get_project_config_path(workspace)
        if project_path is not None:
            with project_path.open("rb") as f:
                project_config = tomllib.load(f)
            raw_config = deep_merge_dict(raw_config, project_config)

    # Resolve secrets from environment
    raw_config = _resolve_env_secrets(raw_config)

    return AshConfig.model_validate(raw_config)


def _find_config_path(path: Path | None = None) -> Path | None:
    """Find config file path."""
    if path is not None:
        config_path = Path(path).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return config_path

    for default_path in _get_default_config_paths():
        expanded = default_path.expanduser()
        if expanded.exists():
            return expanded

    return None


# Example project config at .ash/config.toml:
#
# # Override default model for this project
# [models.default]
# model = "claude-3-5-sonnet-latest"
#
# # Project-specific sandbox settings
# [sandbox]
# python_packages = ["pandas", "numpy"]
#
# # Project-specific environment variables
# [env]
# PROJECT_NAME = "my-project"
```

### Estimated Effort

**Low-Medium** - 2-3 hours

- Implement `deep_merge_dict` function
- Add project config path resolution
- Update `load_config` to merge configs
- Add documentation for project config format

### Priority

**High** - Essential for users working on multiple projects with different requirements (different models, sandbox packages, etc.).

---

## Gap 3: JSON Schema Export

### What Ash is Missing

Ash uses Pydantic models for configuration validation but does not export a JSON Schema that editors/UIs can use for autocompletion and validation:

```python
# models.py - No schema export functionality
class AshConfig(BaseModel):
    """Root configuration model."""
    # ... fields defined but no export
```

### Reference Implementation (Clawdbot)

Clawdbot uses Zod schemas with JSON Schema export capability for editor integration:

```typescript
// zod-schema.ts lines 1110-1555
export const ClawdbotSchema = z
  .object({
    env: z.object({...}).optional(),
    wizard: z.object({...}).optional(),
    logging: z.object({...}).optional(),
    // ... comprehensive schema definition
  })
  .superRefine((cfg, ctx) => {
    // Custom validation rules
  });

// The schema can be converted to JSON Schema for editors:
// import { zodToJsonSchema } from "zod-to-json-schema";
// const jsonSchema = zodToJsonSchema(ClawdbotSchema);
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/config/models.py`
- `/home/dcramer/src/ash/src/ash/config/schema.py` (new file)
- `/home/dcramer/src/ash/src/ash/cli/commands/config.py`

### Concrete Python Code Changes

```python
# config/schema.py - New file for schema export

import json
from pathlib import Path
from typing import Any

from ash.config.models import AshConfig


def get_config_json_schema() -> dict[str, Any]:
    """Generate JSON Schema for configuration.

    Uses Pydantic's built-in schema generation with customizations
    for better editor integration.
    """
    schema = AshConfig.model_json_schema(
        mode="serialization",
        ref_template="#/$defs/{model}",
    )

    # Add schema metadata for editors
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://ash.dev/schemas/config.json"
    schema["title"] = "Ash Configuration"
    schema["description"] = "Configuration schema for Ash personal assistant"

    # Add TOML-specific hints (for editors that support them)
    schema["x-toml-type"] = "document"

    return schema


def export_json_schema(output_path: Path | None = None) -> str:
    """Export JSON Schema to file or return as string.

    Args:
        output_path: If provided, write schema to file. Otherwise return JSON string.

    Returns:
        JSON Schema as formatted string.
    """
    schema = get_config_json_schema()
    schema_json = json.dumps(schema, indent=2)

    if output_path is not None:
        output_path.write_text(schema_json)

    return schema_json


# cli/commands/config.py - Add schema export command

def register(app: typer.Typer) -> None:
    """Register the config command."""

    @app.command()
    def config(
        action: Annotated[
            str,
            typer.Argument(help="Action: show, validate, schema"),
        ],
        path: Annotated[
            Path | None,
            typer.Option(
                "--path",
                "-p",
                help="Path to config file (default: $ASH_HOME/config.toml)",
            ),
        ] = None,
        output: Annotated[
            Path | None,
            typer.Option(
                "--output",
                "-o",
                help="Output path for schema export",
            ),
        ] = None,
    ) -> None:
        """Manage configuration."""
        # ... existing show/validate actions ...

        elif action == "schema":
            from ash.config.schema import export_json_schema

            schema_json = export_json_schema(output)

            if output:
                success(f"Schema exported to {output}")
            else:
                # Print to stdout for piping
                console.print(schema_json)


# config/models.py - Add schema customizations via Field

from pydantic import Field


class ModelConfig(BaseModel):
    """Configuration for a named model."""

    provider: Literal["anthropic", "openai"] = Field(
        ...,
        description="LLM provider to use",
        json_schema_extra={"examples": ["anthropic", "openai"]},
    )
    model: str = Field(
        ...,
        description="Model identifier",
        json_schema_extra={"examples": ["claude-3-5-sonnet-latest", "gpt-4o"]},
    )
    temperature: float | None = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (None = provider default)",
    )
    max_tokens: int = Field(
        4096,
        gt=0,
        description="Maximum tokens in response",
    )
    thinking: Literal["off", "minimal", "low", "medium", "high"] | None = Field(
        None,
        description="Extended thinking level (Anthropic Claude only)",
    )
```

Usage example for editors:

```json
// .vscode/settings.json
{
  "json.schemas": [
    {
      "fileMatch": ["**/config.toml"],
      "url": "./ash-config-schema.json"
    }
  ]
}
```

For TOML support with taplo:

```toml
# .taplo.toml
[schema]
url = "./ash-config-schema.json"
path = "**/config.toml"
```

### Estimated Effort

**Low** - 1-2 hours

- Create `schema.py` with export function
- Add CLI command for schema export
- Add Field descriptions to models
- Document editor integration

### Priority

**Low** - Nice-to-have for power users. Most users edit config manually with documentation.

---

## Gap 4: Atomic Writes with Backup

### What Ash is Missing

Ash's config loader only reads configuration; there is no write functionality. If config writing were added, it would need atomic write support to prevent corruption.

### Reference Implementation (Clawdbot)

Clawdbot uses atomic writes with backup for configuration persistence:

```typescript
// io.ts lines 294-337
async function writeConfigFile(cfg: ClawdbotConfig) {
  const dir = path.dirname(configPath);
  await deps.fs.promises.mkdir(dir, { recursive: true, mode: 0o700 });
  const json = JSON.stringify(applyModelDefaults(cfg), null, 2)
    .trimEnd()
    .concat("\n");

  // Write to temp file first
  const tmp = path.join(
    dir,
    `${path.basename(configPath)}.${process.pid}.${crypto.randomUUID()}.tmp`,
  );

  await deps.fs.promises.writeFile(tmp, json, {
    encoding: "utf-8",
    mode: 0o600,
  });

  // Create backup (best-effort)
  await deps.fs.promises
    .copyFile(configPath, `${configPath}.bak`)
    .catch(() => {
      // best-effort
    });

  try {
    // Atomic rename
    await deps.fs.promises.rename(tmp, configPath);
  } catch (err) {
    const code = (err as { code?: string }).code;
    // Windows fallback: copy instead of rename
    if (code === "EPERM" || code === "EEXIST") {
      await deps.fs.promises.copyFile(tmp, configPath);
      await deps.fs.promises.chmod(configPath, 0o600).catch(() => {});
      await deps.fs.promises.unlink(tmp).catch(() => {});
      return;
    }
    await deps.fs.promises.unlink(tmp).catch(() => {});
    throw err;
  }
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/config/loader.py`

### Concrete Python Code Changes

```python
# config/loader.py - Add atomic write functionality

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import tomlkit  # pip install tomlkit (for TOML writing with formatting preservation)

from ash.config.models import AshConfig


def save_config(
    config: AshConfig,
    path: Path | None = None,
    create_backup: bool = True,
) -> None:
    """Save configuration to TOML file atomically.

    Uses write-to-temp-then-rename pattern to prevent corruption.
    Optionally creates a .bak backup of the previous config.

    Args:
        config: Configuration to save.
        path: Path to config file. If None, uses default config path.
        create_backup: Whether to create .bak backup file.

    Raises:
        IOError: If write fails.
    """
    if path is None:
        path = get_config_path()

    path = path.expanduser()

    # Ensure parent directory exists with secure permissions
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    # Convert config to TOML
    # Exclude None values and use dict for serialization
    config_dict = config.model_dump(
        exclude_none=True,
        exclude_defaults=False,
        mode="json",  # Serialize SecretStr as strings
    )

    # Handle SecretStr values - convert to plain strings for serialization
    config_dict = _mask_secrets_for_display(config_dict)

    toml_content = tomlkit.dumps(config_dict)

    # Write to temp file first
    tmp_path = path.parent / f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"

    try:
        # Write with secure permissions
        tmp_path.write_text(toml_content, encoding="utf-8")
        tmp_path.chmod(0o600)

        # Create backup if config exists
        if create_backup and path.exists():
            backup_path = path.with_suffix(path.suffix + ".bak")
            try:
                shutil.copy2(path, backup_path)
            except OSError:
                # Best-effort backup
                pass

        # Atomic rename (POSIX guarantees atomicity for same-filesystem rename)
        tmp_path.rename(path)

    except Exception:
        # Clean up temp file on failure
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def restore_config_backup(path: Path | None = None) -> AshConfig:
    """Restore configuration from backup file.

    Args:
        path: Path to config file. If None, uses default config path.

    Returns:
        Restored AshConfig instance.

    Raises:
        FileNotFoundError: If backup file doesn't exist.
    """
    if path is None:
        path = get_config_path()

    backup_path = path.with_suffix(path.suffix + ".bak")

    if not backup_path.exists():
        raise FileNotFoundError(f"No backup file found: {backup_path}")

    # Copy backup to main config
    shutil.copy2(backup_path, path)

    return load_config(path)


def _mask_secrets_for_display(config_dict: dict) -> dict:
    """Recursively process config dict to handle secret values.

    Note: This converts SecretStr to their string values for serialization.
    API keys should typically be stored in environment variables, not config files.
    """
    result = {}
    for key, value in config_dict.items():
        if isinstance(value, dict):
            result[key] = _mask_secrets_for_display(value)
        elif key in ("api_key", "bot_token", "dsn") and value:
            # Keep the actual value for writing - user chose to store in config
            result[key] = value
        else:
            result[key] = value
    return result


# Add restore command to CLI
# cli/commands/config.py

elif action == "restore":
    from ash.config.loader import restore_config_backup

    try:
        config_obj = restore_config_backup(expanded_path)
        success("Configuration restored from backup")
    except FileNotFoundError as e:
        error(str(e))
        raise typer.Exit(1)
```

### Estimated Effort

**Low** - 2 hours

- Add `tomlkit` dependency for TOML writing
- Implement `save_config` with atomic write
- Implement `restore_config_backup`
- Add CLI restore command

### Priority

**Medium** - Important if config programmatic modification is needed (e.g., wizard, migration tools).

---

## Gap 5: Shell Environment Fallback

### What Ash is Missing

Ash only reads environment variables directly from `os.environ`. It cannot access environment variables from the user's shell profile when running in headless/daemon mode:

```python
# models.py lines 328-333
# Check environment variable
env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
env_value = os.environ.get(env_var)
if env_value:
    return SecretStr(env_value)
```

### Reference Implementation (Clawdbot)

Clawdbot has a shell environment fallback that spawns a login shell to extract environment variables:

```typescript
// io.ts lines 35-50
const SHELL_ENV_EXPECTED_KEYS = [
  "OPENAI_API_KEY",
  "ANTHROPIC_API_KEY",
  "ANTHROPIC_OAUTH_TOKEN",
  "GEMINI_API_KEY",
  // ... more keys
];

// io.ts lines 136-148, 180-194
function loadConfig(): ClawdbotConfig {
  if (!deps.fs.existsSync(configPath)) {
    if (shouldEnableShellEnvFallback(deps.env)) {
      loadShellEnvFallback({
        enabled: true,
        env: deps.env,
        expectedKeys: SHELL_ENV_EXPECTED_KEYS,
        logger: deps.logger,
        timeoutMs: resolveShellEnvFallbackTimeoutMs(deps.env),
      });
    }
    return {};
  }
  // ...
  const enabled =
    shouldEnableShellEnvFallback(deps.env) ||
    cfg.env?.shellEnv?.enabled === true;
  if (enabled) {
    loadShellEnvFallback({
      enabled: true,
      env: deps.env,
      expectedKeys: SHELL_ENV_EXPECTED_KEYS,
      // ...
    });
  }
}
```

The schema supports explicit configuration (zod-schema.ts lines 1112-1123):

```typescript
env: z
  .object({
    shellEnv: z
      .object({
        enabled: z.boolean().optional(),
        timeoutMs: z.number().int().nonnegative().optional(),
      })
      .optional(),
    vars: z.record(z.string(), z.string()).optional(),
  })
  .catchall(z.string())
  .optional(),
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/config/loader.py`
- `/home/dcramer/src/ash/src/ash/config/models.py`
- `/home/dcramer/src/ash/src/ash/config/shell_env.py` (new file)

### Concrete Python Code Changes

```python
# config/shell_env.py - New file for shell environment fallback

import asyncio
import logging
import os
import shutil
import subprocess
from typing import Sequence

logger = logging.getLogger(__name__)

# Environment variables we want to extract from user's shell
SHELL_ENV_EXPECTED_KEYS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "BRAVE_SEARCH_API_KEY",
    "SENTRY_DSN",
]

DEFAULT_TIMEOUT_MS = 5000


def should_enable_shell_env_fallback() -> bool:
    """Check if shell env fallback should be enabled.

    Enabled when:
    - Running as a daemon/service (no TTY)
    - ASH_SHELL_ENV_FALLBACK=1 is set
    - Not explicitly disabled via ASH_SHELL_ENV_FALLBACK=0
    """
    explicit = os.environ.get("ASH_SHELL_ENV_FALLBACK")
    if explicit is not None:
        return explicit.lower() in ("1", "true", "yes")

    # Enable if running without a TTY (daemon mode)
    return not os.isatty(0)


def get_login_shell() -> str | None:
    """Get the user's login shell."""
    shell = os.environ.get("SHELL")
    if shell and shutil.which(shell):
        return shell

    # Fallback to common shells
    for shell_path in ["/bin/bash", "/bin/zsh", "/bin/sh"]:
        if shutil.which(shell_path):
            return shell_path

    return None


def load_shell_env_fallback(
    expected_keys: Sequence[str] | None = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> dict[str, str]:
    """Load environment variables from user's login shell.

    Spawns a login shell and extracts specified environment variables.
    This is useful when running as a daemon where the parent process
    doesn't have the user's shell environment.

    Args:
        expected_keys: Environment variable names to extract.
        timeout_ms: Timeout in milliseconds.

    Returns:
        Dict of extracted environment variables.
    """
    if expected_keys is None:
        expected_keys = SHELL_ENV_EXPECTED_KEYS

    shell = get_login_shell()
    if shell is None:
        logger.warning("No login shell found for env fallback")
        return {}

    # Build command to print expected env vars
    # Use printf for reliable parsing (handles values with newlines)
    env_script = "; ".join(
        f'printf "%s\\0" "${{{key}:-}}"' for key in expected_keys
    )

    # Run as login shell to source profile
    cmd = [shell, "-l", "-c", env_script]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_ms / 1000,
            text=False,  # Binary mode for null-separated parsing
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"Shell env fallback timed out after {timeout_ms}ms")
        return {}
    except Exception as e:
        logger.warning(f"Shell env fallback failed: {e}")
        return {}

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        logger.warning(f"Shell env fallback failed: {stderr}")
        return {}

    # Parse null-separated output
    values = result.stdout.split(b"\0")
    env_vars = {}

    for key, value in zip(expected_keys, values):
        decoded = value.decode("utf-8", errors="replace").strip()
        if decoded:
            env_vars[key] = decoded
            # Also set in current process environment
            os.environ[key] = decoded

    if env_vars:
        logger.info(f"Loaded {len(env_vars)} env vars from shell: {list(env_vars.keys())}")

    return env_vars


# config/models.py - Add shell env config

class ShellEnvConfig(BaseModel):
    """Configuration for shell environment fallback."""

    enabled: bool = False  # Explicit enable
    timeout_ms: int = Field(default=5000, ge=0)
    expected_keys: list[str] = Field(default_factory=list)


class EnvConfig(BaseModel):
    """Environment variable configuration."""

    shell_env: ShellEnvConfig = Field(default_factory=ShellEnvConfig)
    vars: dict[str, str] = Field(default_factory=dict)


# Update AshConfig
class AshConfig(BaseModel):
    # ... existing fields ...
    env_config: EnvConfig = Field(default_factory=EnvConfig, alias="env_settings")


# config/loader.py - Integrate shell env fallback

from ash.config.shell_env import (
    load_shell_env_fallback,
    should_enable_shell_env_fallback,
)


def load_config(path: Path | None = None) -> AshConfig:
    # ... existing path resolution ...

    # Load shell env before parsing config (so env vars are available)
    if should_enable_shell_env_fallback():
        load_shell_env_fallback()

    # ... rest of loading logic ...

    # Also check config for explicit shell_env settings
    config = AshConfig.model_validate(raw_config)

    if config.env_config.shell_env.enabled:
        load_shell_env_fallback(
            expected_keys=config.env_config.shell_env.expected_keys or None,
            timeout_ms=config.env_config.shell_env.timeout_ms,
        )

    return config
```

### Estimated Effort

**Low-Medium** - 2-3 hours

- Create `shell_env.py` module
- Add config schema for shell_env settings
- Integrate into loader
- Test with systemd service

### Priority

**Medium** - Important for production deployments where Ash runs as a system service.

---

## Gap 6: Enhanced Config Validation CLI

### What Ash is Missing

Ash has basic config validation that shows Pydantic errors:

```python
# cli/commands/config.py lines 51-107
elif action == "validate":
    # ... basic validation with generic error display ...
    except ValidationError as e:
        error("Configuration validation failed:")
        console.print()
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            console.print(f"  [yellow]{loc}[/yellow]: {err['msg']}")
```

This lacks:
- Suggestions for common mistakes
- Deprecation warnings with migration guidance
- Detection of mistyped keys
- Detailed context for complex errors

### Reference Implementation (Clawdbot)

Clawdbot has comprehensive validation with legacy detection and detailed feedback:

```typescript
// io.ts lines 65-77 - Miskey warnings
function warnOnConfigMiskeys(
  raw: unknown,
  logger: Pick<typeof console, "warn">,
): void {
  if (!raw || typeof raw !== "object") return;
  const gateway = (raw as Record<string, unknown>).gateway;
  if (!gateway || typeof gateway !== "object") return;
  if ("token" in (gateway as Record<string, unknown>)) {
    logger.warn(
      'Config uses "gateway.token". This key is ignored; use "gateway.auth.token" instead.',
    );
  }
}

// io.ts lines 207-291 - readConfigFileSnapshot with detailed validation
async function readConfigFileSnapshot(): Promise<ConfigFileSnapshot> {
  // ... returns structured result with:
  // - path: config file path
  // - exists: whether file exists
  // - raw: raw file content
  // - parsed: parsed JSON/TOML
  // - valid: validation result
  // - config: validated config object
  // - issues: array of validation issues with path and message
  // - legacyIssues: detected deprecated settings with migration hints
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/cli/commands/config.py`
- `/home/dcramer/src/ash/src/ash/config/validation.py` (new file)

### Concrete Python Code Changes

```python
# config/validation.py - New file for enhanced validation

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib
from pydantic import ValidationError

from ash.config.models import AshConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A validation issue with path and message."""
    path: str
    message: str
    severity: str = "error"  # error, warning, info
    suggestion: str | None = None


@dataclass
class LegacyIssue:
    """A deprecated configuration detected."""
    path: str
    message: str
    migration: str  # How to migrate


@dataclass
class ConfigSnapshot:
    """Complete config file analysis result."""
    config_path: Path
    exists: bool
    raw: str | None
    parsed: dict[str, Any] | None
    valid: bool
    config: AshConfig | None
    issues: list[ValidationIssue]
    legacy_issues: list[LegacyIssue]
    warnings: list[str]


# Known mistyped keys and their corrections
MISKEY_CORRECTIONS = {
    ("default_model",): ("models", "default"),
    ("llm",): ("models", "default"),
    ("model",): ("models", "default"),
    ("api_key",): ("anthropic", "api_key"),
    ("openai_api_key",): ("openai", "api_key"),
    ("anthropic_api_key",): ("anthropic", "api_key"),
    ("telegram_token",): ("telegram", "bot_token"),
    ("sandbox", "docker_image"): ("sandbox", "image"),
    ("memory", "db_path"): ("memory", "database_path"),
}

# Deprecated settings and migration guidance
DEPRECATED_SETTINGS = {
    ("default_llm",): LegacyIssue(
        path="default_llm",
        message="[default_llm] is deprecated",
        migration="Use [models.default] instead:\n"
                  "[models.default]\n"
                  "provider = \"anthropic\"\n"
                  "model = \"claude-3-5-sonnet-latest\"",
    ),
    ("fallback_llm",): LegacyIssue(
        path="fallback_llm",
        message="[fallback_llm] is deprecated",
        migration="Use named models with fallback logic in your code:\n"
                  "[models.fallback]\n"
                  "provider = \"openai\"\n"
                  "model = \"gpt-4o\"",
    ),
}


def find_miskeys(parsed: dict[str, Any]) -> list[ValidationIssue]:
    """Find common mistyped configuration keys."""
    issues = []

    def check_dict(d: dict, path: tuple = ()) -> None:
        for key, value in d.items():
            current_path = path + (key,)

            if current_path in MISKEY_CORRECTIONS:
                correct = ".".join(MISKEY_CORRECTIONS[current_path])
                issues.append(ValidationIssue(
                    path=".".join(current_path),
                    message=f"Unknown key '{key}'",
                    severity="warning",
                    suggestion=f"Did you mean '{correct}'?",
                ))

            if isinstance(value, dict):
                check_dict(value, current_path)

    check_dict(parsed)
    return issues


def find_legacy_settings(parsed: dict[str, Any]) -> list[LegacyIssue]:
    """Find deprecated configuration settings."""
    issues = []

    def check_dict(d: dict, path: tuple = ()) -> None:
        for key, value in d.items():
            current_path = path + (key,)

            if current_path in DEPRECATED_SETTINGS:
                issues.append(DEPRECATED_SETTINGS[current_path])

            if isinstance(value, dict):
                check_dict(value, current_path)

    check_dict(parsed)
    return issues


def validate_config_file(path: Path) -> ConfigSnapshot:
    """Perform comprehensive config file validation.

    Returns detailed analysis including:
    - Parse errors
    - Validation errors with context
    - Deprecated setting warnings
    - Mistyped key suggestions
    """
    if not path.exists():
        return ConfigSnapshot(
            config_path=path,
            exists=False,
            raw=None,
            parsed=None,
            valid=False,
            config=None,
            issues=[ValidationIssue(
                path="",
                message=f"Config file not found: {path}",
            )],
            legacy_issues=[],
            warnings=[],
        )

    # Read raw content
    try:
        raw = path.read_text()
    except Exception as e:
        return ConfigSnapshot(
            config_path=path,
            exists=True,
            raw=None,
            parsed=None,
            valid=False,
            config=None,
            issues=[ValidationIssue(
                path="",
                message=f"Failed to read file: {e}",
            )],
            legacy_issues=[],
            warnings=[],
        )

    # Parse TOML
    try:
        parsed = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as e:
        return ConfigSnapshot(
            config_path=path,
            exists=True,
            raw=raw,
            parsed=None,
            valid=False,
            config=None,
            issues=[ValidationIssue(
                path="",
                message=f"TOML parse error: {e}",
                suggestion="Check for syntax errors like missing quotes or brackets",
            )],
            legacy_issues=[],
            warnings=[],
        )

    # Check for miskeys and legacy settings (before validation)
    miskey_issues = find_miskeys(parsed)
    legacy_issues = find_legacy_settings(parsed)
    warnings = []

    # Validate with Pydantic
    try:
        from ash.config.loader import _resolve_env_secrets
        resolved = _resolve_env_secrets(parsed.copy())
        config = AshConfig.model_validate(resolved)

        return ConfigSnapshot(
            config_path=path,
            exists=True,
            raw=raw,
            parsed=parsed,
            valid=True,
            config=config,
            issues=miskey_issues,  # Warnings only
            legacy_issues=legacy_issues,
            warnings=warnings,
        )
    except ValidationError as e:
        validation_issues = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])

            # Add suggestions for common errors
            suggestion = None
            if "missing" in err["type"]:
                suggestion = f"Add the required field to your config"
            elif "enum" in err["type"]:
                suggestion = f"Valid values: {err.get('ctx', {}).get('expected', 'see documentation')}"

            validation_issues.append(ValidationIssue(
                path=loc,
                message=err["msg"],
                suggestion=suggestion,
            ))

        return ConfigSnapshot(
            config_path=path,
            exists=True,
            raw=raw,
            parsed=parsed,
            valid=False,
            config=None,
            issues=miskey_issues + validation_issues,
            legacy_issues=legacy_issues,
            warnings=warnings,
        )


# cli/commands/config.py - Enhanced validate action

elif action == "validate":
    from ash.config.validation import validate_config_file
    from rich.panel import Panel

    snapshot = validate_config_file(expanded_path)

    # Show file status
    console.print(f"[bold]Config file:[/bold] {snapshot.config_path}")
    console.print()

    # Show legacy warnings first
    if snapshot.legacy_issues:
        console.print("[yellow bold]Deprecated Settings:[/yellow bold]")
        for issue in snapshot.legacy_issues:
            console.print(Panel(
                f"[yellow]{issue.message}[/yellow]\n\n"
                f"[dim]Migration:[/dim]\n{issue.migration}",
                title=f"[yellow]{issue.path}[/yellow]",
                border_style="yellow",
            ))
        console.print()

    # Show validation issues
    if snapshot.issues:
        has_errors = any(i.severity == "error" for i in snapshot.issues)
        title = "[red bold]Validation Errors:[/red bold]" if has_errors else "[yellow bold]Warnings:[/yellow bold]"
        console.print(title)

        for issue in snapshot.issues:
            color = "red" if issue.severity == "error" else "yellow"
            console.print(f"  [{color}]{issue.path}[/{color}]: {issue.message}")
            if issue.suggestion:
                console.print(f"    [dim]Suggestion: {issue.suggestion}[/dim]")
        console.print()

    # Show result
    if snapshot.valid:
        success("Configuration is valid!")
        # ... show summary table ...
    else:
        error("Configuration validation failed")
        raise typer.Exit(1)
```

### Estimated Effort

**Medium** - 3-4 hours

- Create `validation.py` module
- Define miskey and deprecation mappings
- Implement `validate_config_file`
- Update CLI with rich output
- Test with various config files

### Priority

**Medium** - Improves user experience significantly when debugging config issues.

---

## Gap 7: Config Snapshot API

### What Ash is Missing

Ash has no API endpoint to retrieve the current configuration state. This would be useful for:
- Web UIs displaying current settings
- Debugging tools
- Configuration editors

### Reference Implementation (Clawdbot)

Clawdbot provides a `readConfigFileSnapshot` function that returns complete config state:

```typescript
// io.ts lines 207-291
async function readConfigFileSnapshot(): Promise<ConfigFileSnapshot> {
  const exists = deps.fs.existsSync(configPath);
  if (!exists) {
    const config = applyTalkApiKey(
      applyModelDefaults(
        applyContextPruningDefaults(
          applySessionDefaults(applyMessageDefaults({})),
        ),
      ),
    );
    const legacyIssues: LegacyConfigIssue[] = [];
    return {
      path: configPath,
      exists: false,
      raw: null,
      parsed: {},
      valid: true,
      config,
      issues: [],
      legacyIssues,
    };
  }

  try {
    const raw = deps.fs.readFileSync(configPath, "utf-8");
    const parsedRes = parseConfigJson5(raw, deps.json5);
    // ... validation and return

    return {
      path: configPath,
      exists: true,
      raw,
      parsed: parsedRes.parsed,
      valid: true,
      config: /* validated config */,
      issues: [],
      legacyIssues,
    };
  } catch (err) {
    return {
      path: configPath,
      exists: true,
      raw: null,
      parsed: {},
      valid: false,
      config: {},
      issues: [{ path: "", message: `read failed: ${String(err)}` }],
      legacyIssues: [],
    };
  }
}

// Exported for API use
export async function readConfigFileSnapshot(): Promise<ConfigFileSnapshot> {
  return await createConfigIO({
    configPath: resolveConfigPath(),
  }).readConfigFileSnapshot();
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/config/loader.py`
- `/home/dcramer/src/ash/src/ash/server/routes/config.py` (new file)
- `/home/dcramer/src/ash/src/ash/server/app.py`

### Concrete Python Code Changes

```python
# config/loader.py - Add snapshot function

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ConfigFileSnapshot:
    """Complete snapshot of configuration file state."""
    path: str
    exists: bool
    raw: str | None
    parsed: dict[str, Any] | None
    valid: bool
    config: dict[str, Any] | None  # Serialized config (secrets masked)
    issues: list[dict[str, str]]
    legacy_issues: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)


def read_config_snapshot(path: Path | None = None) -> ConfigFileSnapshot:
    """Read config file and return complete snapshot.

    Used by APIs and debugging tools to inspect configuration state.
    Secrets are masked in the returned config.
    """
    from ash.config.validation import validate_config_file

    if path is None:
        path = get_config_path()

    snapshot = validate_config_file(path)

    # Mask secrets in config for API response
    config_dict = None
    if snapshot.config is not None:
        config_dict = snapshot.config.model_dump(
            exclude_none=True,
            mode="json",
        )
        config_dict = _mask_secrets(config_dict)

    return ConfigFileSnapshot(
        path=str(snapshot.config_path),
        exists=snapshot.exists,
        raw=snapshot.raw,
        parsed=snapshot.parsed,
        valid=snapshot.valid,
        config=config_dict,
        issues=[
            {"path": i.path, "message": i.message, "suggestion": i.suggestion or ""}
            for i in snapshot.issues
        ],
        legacy_issues=[
            {"path": i.path, "message": i.message, "migration": i.migration}
            for i in snapshot.legacy_issues
        ],
    )


def _mask_secrets(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Recursively mask secret values in config dict."""
    SECRET_KEYS = {"api_key", "bot_token", "dsn", "password", "secret"}

    result = {}
    for key, value in config_dict.items():
        if isinstance(value, dict):
            result[key] = _mask_secrets(value)
        elif key in SECRET_KEYS and value:
            # Show first/last chars for identification
            if len(value) > 8:
                result[key] = f"{value[:4]}...{value[-4:]}"
            else:
                result[key] = "***"
        else:
            result[key] = value
    return result


# server/routes/config.py - New API route

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ash.config.loader import read_config_snapshot, ConfigFileSnapshot

router = APIRouter(prefix="/config", tags=["config"])


class ConfigSnapshotResponse(BaseModel):
    """API response for config snapshot."""
    path: str
    exists: bool
    raw: str | None
    parsed: dict | None
    valid: bool
    config: dict | None
    issues: list[dict]
    legacy_issues: list[dict]


@router.get("/snapshot", response_model=ConfigSnapshotResponse)
async def get_config_snapshot() -> ConfigSnapshotResponse:
    """Get current configuration file snapshot.

    Returns the complete state of the configuration file including:
    - Raw file content
    - Parsed structure
    - Validation status
    - Current config (with secrets masked)
    - Any validation issues or deprecation warnings
    """
    snapshot = read_config_snapshot()
    return ConfigSnapshotResponse(**snapshot.to_dict())


@router.get("/current")
async def get_current_config() -> dict:
    """Get the currently active configuration (secrets masked).

    Returns just the validated config without file metadata.
    """
    snapshot = read_config_snapshot()
    if not snapshot.valid or snapshot.config is None:
        raise HTTPException(
            status_code=500,
            detail="Configuration is invalid",
        )
    return snapshot.config


# server/app.py - Register route

from ash.server.routes import config as config_routes

app.include_router(config_routes.router)
```

### Estimated Effort

**Low-Medium** - 2-3 hours

- Add `read_config_snapshot` function
- Create API routes
- Add secret masking
- Register routes in app

### Priority

**Low** - Useful for future web UI or debugging tools, but not critical for core functionality.

---

## Summary

| Gap | Description | Effort | Priority |
|-----|-------------|--------|----------|
| 1. Hot Config Reload | Watch config file and reload without restart | Medium (3-4h) | Medium |
| 2. Per-Project Settings | .ash/config.toml in workspace with deep merge | Low-Medium (2-3h) | **High** |
| 3. JSON Schema Export | Export schema for editor autocompletion | Low (1-2h) | Low |
| 4. Atomic Writes | Temp file + rename + backup for safe writes | Low (2h) | Medium |
| 5. Shell Env Fallback | Extract env vars from login shell for daemons | Low-Medium (2-3h) | Medium |
| 6. Enhanced Validation CLI | Detailed errors, suggestions, deprecation warnings | Medium (3-4h) | Medium |
| 7. Config Snapshot API | API endpoint for current config state | Low-Medium (2-3h) | Low |

### Recommended Implementation Order

1. **Per-Project Settings** (Gap 2) - High value for multi-project users
2. **Shell Env Fallback** (Gap 5) - Important for production deployments
3. **Enhanced Validation CLI** (Gap 6) - Better user experience
4. **Atomic Writes** (Gap 4) - Safety for config modifications
5. **Hot Config Reload** (Gap 1) - Developer convenience
6. **JSON Schema Export** (Gap 3) - Editor integration
7. **Config Snapshot API** (Gap 7) - Future web UI support
