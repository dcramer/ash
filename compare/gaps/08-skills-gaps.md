# Skills System Gap Analysis

This document analyzes 8 specific gaps between ash's skills implementation and the reference implementations in clawdbot and pi-mono.

## Files Analyzed

**Ash:**
- `/home/dcramer/src/ash/src/ash/skills/base.py` - Skill dataclasses and requirements
- `/home/dcramer/src/ash/src/ash/skills/registry.py` - Skill discovery and loading
- `/home/dcramer/src/ash/src/ash/skills/state.py` - Persistent skill state storage

**References:**
- `/home/dcramer/src/clawdbot/src/agents/skills.ts` - Most sophisticated skill system
- `/home/dcramer/src/pi-mono/packages/coding-agent/src/core/skills.ts` - Multi-source loading with validation

---

## Gap 1: anyBins Requirement Check

### What Ash is Missing

Ash's `SkillRequirements.bins` field requires **all** listed binaries to exist. There's no way to express "at least one of these binaries must exist" - a common pattern for tools with multiple implementations (e.g., `npm` OR `pnpm` OR `yarn`).

Current ash code (`base.py` lines 52-55):
```python
# Check binaries
for bin_name in self.bins:
    if not shutil.which(bin_name):
        return False, f"Requires binary: {bin_name}"
```

Clawdbot has both `bins` (all required) and `anyBins` (at least one required):
```typescript
requires?: {
  bins?: string[];      // All must exist
  anyBins?: string[];   // At least one must exist
  env?: string[];
  config?: string[];
};
```

### Reference

**Best implementation:** clawdbot (`skills.ts` lines 350-360)
```typescript
const requiredAnyBins = entry.clawdbot?.requires?.anyBins ?? [];
if (requiredAnyBins.length > 0) {
  const anyFound = requiredAnyBins.some((bin) => hasBinary(bin));
  if (!anyFound) return false;
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/skills/base.py`

### Proposed Changes

```python
# In base.py, modify SkillRequirements dataclass:

@dataclass
class SkillRequirements:
    """Requirements for a skill to be available.

    Skills are filtered out if requirements aren't met.
    """

    # Required binaries (all must exist in PATH)
    bins: list[str] = field(default_factory=list)

    # Alternative binaries (at least one must exist in PATH)
    # Use for tools with multiple implementations (npm/pnpm/yarn)
    any_bins: list[str] = field(default_factory=list)

    # Required environment variables (all must be set)
    env: list[str] = field(default_factory=list)

    # ... rest unchanged ...

    def check(self) -> tuple[bool, str | None]:
        """Check if all requirements are met."""
        # Check OS
        if self.os:
            current_os = platform.system().lower()
            if current_os not in self.os:
                return (
                    False,
                    f"Requires OS: {', '.join(self.os)} (current: {current_os})",
                )

        # Check binaries (all must exist)
        for bin_name in self.bins:
            if not shutil.which(bin_name):
                return False, f"Requires binary: {bin_name}"

        # Check any_bins (at least one must exist)
        if self.any_bins:
            found = any(shutil.which(bin_name) for bin_name in self.any_bins)
            if not found:
                return False, f"Requires one of: {', '.join(self.any_bins)}"

        # Check environment variables
        for env_var in self.env:
            if not os.environ.get(env_var):
                return False, f"Requires environment variable: {env_var}"

        return True, None
```

```python
# In registry.py, modify _parse_requirements:

def _parse_requirements(self, data: dict[str, Any]) -> SkillRequirements:
    """Parse requirements from skill data."""
    requires = data.get("requires", {})
    if not isinstance(requires, dict):
        return SkillRequirements()

    return SkillRequirements(
        bins=requires.get("bins", []),
        any_bins=requires.get("any_bins", []),  # Add this line
        env=requires.get("env", []),
        os=requires.get("os", []),
        apt_packages=requires.get("apt_packages", []),
        python_packages=requires.get("python_packages", []),
        python_tools=requires.get("python_tools", []),
    )
```

### Effort

**S** (1-2 hours) - Simple dataclass field addition and one conditional check.

### Priority

**Medium** - Useful for package manager detection but not critical for most skills.

---

## Gap 2: Multi-Source Skill Loading

### What Ash is Missing

Ash only loads skills from a single location: `workspace/skills/`. Pi-mono loads from multiple sources with precedence:
1. `~/.codex/skills/` (Codex user)
2. `~/.claude/skills/` (Claude user)
3. `<project>/.claude/skills/` (Claude project)
4. `~/.pi/agent/skills/` (Pi user)
5. `<project>/.pi/skills/` (Pi project)
6. Custom directories from config

This means users can't have global skills that work across all projects, and can't layer bundled skills with workspace overrides.

Current ash code (`registry.py` lines 39-56):
```python
def discover(self, workspace_path: Path, *, include_bundled: bool = True) -> None:
    """Load skills from workspace directory."""
    skills_dir = workspace_path / "skills"
    if not skills_dir.exists():
        logger.debug(f"Workspace skills directory not found: {skills_dir}")
        return

    self._load_from_directory(skills_dir, source="workspace")
```

### Reference

**Best implementation:** pi-mono (`skills.ts` lines 342-436)
```typescript
export function loadSkills(options: LoadSkillsOptions = {}): LoadSkillsResult {
  const {
    cwd = process.cwd(),
    enableCodexUser = true,
    enableClaudeUser = true,
    enableClaudeProject = true,
    enablePiUser = true,
    enablePiProject = true,
    customDirectories = [],
    // ...
  } = options;

  // Load from each source with precedence handling
  if (enableCodexUser) {
    addSkills(loadSkillsFromDirInternal(join(homedir(), ".codex", "skills"), "codex-user", "recursive"));
  }
  if (enableClaudeUser) {
    addSkills(loadSkillsFromDirInternal(join(homedir(), ".claude", "skills"), "claude-user", "claude"));
  }
  // ... etc
}
```

Also see clawdbot (`skills.ts` lines 486-544):
```typescript
const bundledSkills = bundledSkillsDir ? loadSkills({dir: bundledSkillsDir, source: "clawdbot-bundled"}) : [];
const extraSkills = extraDirs.flatMap((dir) => loadSkills({dir: resolved, source: "clawdbot-extra"}));
const managedSkills = loadSkills({dir: managedSkillsDir, source: "clawdbot-managed"});
const workspaceSkills = loadSkills({dir: workspaceSkillsDir, source: "clawdbot-workspace"});

// Precedence: extra < bundled < managed < workspace
for (const skill of extraSkills) merged.set(skill.name, skill);
for (const skill of bundledSkills) merged.set(skill.name, skill);
for (const skill of managedSkills) merged.set(skill.name, skill);
for (const skill of workspaceSkills) merged.set(skill.name, skill);
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/skills/registry.py`
- `/home/dcramer/src/ash/src/ash/config/models.py` (add skills config section)

### Proposed Changes

```python
# Add to config/models.py:

class SkillsConfig(BaseModel):
    """Configuration for skills system."""

    # Enable loading from various sources
    enable_user_skills: bool = True  # ~/.ash/skills/
    enable_workspace_skills: bool = True  # <workspace>/skills/

    # Additional directories to load skills from
    extra_dirs: list[str] = Field(default_factory=list)


# Add to AshConfig class:
class AshConfig(BaseModel):
    # ... existing fields ...
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
```

```python
# Modify registry.py:

from pathlib import Path
from ash.config.paths import get_ash_home

class SkillRegistry:
    """Registry for skill definitions.

    Loads skills from multiple sources with precedence:
    1. Extra directories (lowest)
    2. User skills (~/.ash/skills/)
    3. Workspace skills (highest - can override)
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._skills: dict[str, SkillDefinition] = {}
        self._skill_sources: dict[str, Path] = {}

    def discover(
        self,
        workspace_path: Path,
        *,
        enable_user_skills: bool = True,
        enable_workspace_skills: bool = True,
        extra_dirs: list[str] | None = None,
    ) -> None:
        """Load skills from all configured sources.

        Precedence (later sources override earlier):
        1. Extra directories from config
        2. User skills (~/.ash/skills/)
        3. Workspace skills (workspace/skills/)

        Args:
            workspace_path: Path to workspace directory.
            enable_user_skills: Load from ~/.ash/skills/
            enable_workspace_skills: Load from workspace/skills/
            extra_dirs: Additional directories to load from.
        """
        # Load extra directories first (lowest precedence)
        for extra_dir in extra_dirs or []:
            extra_path = Path(extra_dir).expanduser()
            if extra_path.exists():
                self._load_from_directory(extra_path, source="extra")

        # Load user skills
        if enable_user_skills:
            user_skills_dir = get_ash_home() / "skills"
            if user_skills_dir.exists():
                self._load_from_directory(user_skills_dir, source="user")

        # Load workspace skills last (highest precedence)
        if enable_workspace_skills:
            skills_dir = workspace_path / "skills"
            if skills_dir.exists():
                self._load_from_directory(skills_dir, source="workspace")
```

```python
# In paths.py, add:

def get_ash_home() -> Path:
    """Get the ash home directory (~/.ash)."""
    return Path.home() / ".ash"
```

### Effort

**M** (half day) - Requires config schema changes and refactoring discover() with new parameters.

### Priority

**High** - Users need global skills that work across projects. This is table stakes for a good skill system.

---

## Gap 3: Glob-Based Filtering

### What Ash is Missing

Ash has no way to filter which skills are loaded. Pi-mono supports:
- `ignoredSkills`: Glob patterns to exclude skills
- `includeSkills`: Glob patterns to include only matching skills

This is useful for:
- Disabling a problematic skill without deleting it
- Having different skill sets for different projects
- Temporarily focusing on specific skills during development

### Reference

**Best implementation:** pi-mono (`skills.ts` lines 364-385)
```typescript
function matchesIncludePatterns(name: string): boolean {
  if (includeSkills.length === 0) return true; // No filter = include all
  return includeSkills.some((pattern) => minimatch(name, pattern));
}

function matchesIgnorePatterns(name: string): boolean {
  if (ignoredSkills.length === 0) return false;
  return ignoredSkills.some((pattern) => minimatch(name, pattern));
}

function addSkills(result: LoadSkillsResult) {
  for (const skill of result.skills) {
    // Apply ignore filter (glob patterns) - takes precedence over include
    if (matchesIgnorePatterns(skill.name)) {
      continue;
    }
    // Apply include filter (glob patterns)
    if (!matchesIncludePatterns(skill.name)) {
      continue;
    }
    // ... rest of logic
  }
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/config/models.py`
- `/home/dcramer/src/ash/src/ash/skills/registry.py`

### Proposed Changes

```python
# In config/models.py, modify SkillsConfig:

class SkillsConfig(BaseModel):
    """Configuration for skills system."""

    enable_user_skills: bool = True
    enable_workspace_skills: bool = True
    extra_dirs: list[str] = Field(default_factory=list)

    # Glob patterns for filtering
    ignored_skills: list[str] = Field(default_factory=list)  # Exclude matching
    include_skills: list[str] = Field(default_factory=list)  # Include only matching
```

```python
# In registry.py, add filtering:

import fnmatch


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}
        self._skill_sources: dict[str, Path] = {}
        self._ignored_patterns: list[str] = []
        self._include_patterns: list[str] = []

    def set_filters(
        self,
        ignored_skills: list[str] | None = None,
        include_skills: list[str] | None = None,
    ) -> None:
        """Set glob patterns for skill filtering.

        Args:
            ignored_skills: Patterns to exclude (takes precedence).
            include_skills: Patterns to include (empty = include all).
        """
        self._ignored_patterns = ignored_skills or []
        self._include_patterns = include_skills or []

    def _should_include_skill(self, name: str) -> bool:
        """Check if a skill should be included based on filters."""
        # Ignore patterns take precedence
        for pattern in self._ignored_patterns:
            if fnmatch.fnmatch(name, pattern):
                logger.debug(f"Skill '{name}' excluded by ignore pattern '{pattern}'")
                return False

        # If no include patterns, include everything not ignored
        if not self._include_patterns:
            return True

        # Must match at least one include pattern
        for pattern in self._include_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True

        logger.debug(f"Skill '{name}' excluded (no include pattern matched)")
        return False

    def _register_skill(self, skill: SkillDefinition, source_path: Path) -> None:
        """Register a skill, logging warnings for overrides."""
        # Apply filters before registration
        if not self._should_include_skill(skill.name):
            return

        if skill.name in self._skills:
            existing_source = self._skill_sources.get(skill.name)
            if existing_source and existing_source != source_path:
                logger.warning(f"Skill '{skill.name}' overwritten by {source_path}")

        # Check availability and log if not available
        is_available, reason = skill.is_available()
        if not is_available:
            logger.debug(f"Skill '{skill.name}' not available: {reason}")

        self._skills[skill.name] = skill
        self._skill_sources[skill.name] = source_path
        logger.debug(f"Loaded skill: {skill.name} from {source_path}")
```

### Effort

**S** (1-2 hours) - Simple pattern matching with fnmatch (stdlib).

### Priority

**Low** - Nice to have but not essential. Most users won't need fine-grained filtering.

---

## Gap 4: Install Spec Documentation

### What Ash is Missing

Clawdbot skills can document how to install missing dependencies with structured install specs:

```typescript
install?: SkillInstallSpec[];  // Array of installation options

type SkillInstallSpec = {
  id?: string;
  kind: "brew" | "node" | "go" | "uv";
  label?: string;
  bins?: string[];
  formula?: string;
  package?: string;
  module?: string;
};
```

This allows the system to:
1. Tell users exactly how to install missing deps
2. Potentially auto-install (with permission)
3. Support multiple install methods (brew OR npm)

Ash has nothing - just "Requires binary: foo" with no guidance.

### Reference

**Best implementation:** clawdbot (`skills.ts` lines 16-24, 136-163)
```typescript
export type SkillInstallSpec = {
  id?: string;
  kind: "brew" | "node" | "go" | "uv";
  label?: string;
  bins?: string[];
  formula?: string;
  package?: string;
  module?: string;
};

function parseInstallSpec(input: unknown): SkillInstallSpec | undefined {
  // ... parsing logic
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/skills/base.py`
- `/home/dcramer/src/ash/src/ash/skills/registry.py`

### Proposed Changes

```python
# In base.py, add new dataclass:

from typing import Literal

@dataclass
class InstallSpec:
    """Installation specification for a skill dependency.

    Describes how to install a missing binary or package.
    """

    kind: Literal["brew", "pip", "uv", "apt", "npm", "go"]

    # Human-readable label (e.g., "Homebrew", "pip install")
    label: str | None = None

    # For brew: formula name (e.g., "jq")
    formula: str | None = None

    # For pip/npm: package name
    package: str | None = None

    # For uv: tool name (for uvx)
    tool: str | None = None

    # For go: module path
    module: str | None = None

    # Binaries this install provides
    bins: list[str] = field(default_factory=list)

    def get_install_command(self) -> str:
        """Get the shell command to install this dependency."""
        if self.kind == "brew" and self.formula:
            return f"brew install {self.formula}"
        elif self.kind == "pip" and self.package:
            return f"pip install {self.package}"
        elif self.kind == "uv" and self.tool:
            return f"uvx install {self.tool}"
        elif self.kind == "apt" and self.package:
            return f"sudo apt install {self.package}"
        elif self.kind == "npm" and self.package:
            return f"npm install -g {self.package}"
        elif self.kind == "go" and self.module:
            return f"go install {self.module}@latest"
        return f"# Install via {self.kind}: {self.package or self.formula or self.tool or self.module}"


@dataclass
class SkillRequirements:
    """Requirements for a skill to be available."""

    bins: list[str] = field(default_factory=list)
    any_bins: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    os: list[str] = field(default_factory=list)
    apt_packages: list[str] = field(default_factory=list)
    python_packages: list[str] = field(default_factory=list)
    python_tools: list[str] = field(default_factory=list)

    # Installation instructions for missing dependencies
    install: list[InstallSpec] = field(default_factory=list)

    def check(self) -> tuple[bool, str | None]:
        """Check if all requirements are met."""
        # ... existing check logic ...

    def get_missing_install_instructions(self) -> list[str]:
        """Get installation commands for missing binaries."""
        instructions = []

        # Check which binaries are missing
        missing_bins = [b for b in self.bins if not shutil.which(b)]

        for spec in self.install:
            # If this spec provides a missing binary, include it
            if any(b in missing_bins for b in spec.bins):
                instructions.append(spec.get_install_command())

        return instructions
```

```python
# In registry.py, modify _parse_requirements:

def _parse_install_specs(self, data: list[dict[str, Any]]) -> list[InstallSpec]:
    """Parse install specifications from skill data."""
    specs = []
    for item in data:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind", "").lower()
        if kind not in ("brew", "pip", "uv", "apt", "npm", "go"):
            continue
        specs.append(InstallSpec(
            kind=kind,  # type: ignore
            label=item.get("label"),
            formula=item.get("formula"),
            package=item.get("package"),
            tool=item.get("tool"),
            module=item.get("module"),
            bins=item.get("bins", []),
        ))
    return specs

def _parse_requirements(self, data: dict[str, Any]) -> SkillRequirements:
    """Parse requirements from skill data."""
    requires = data.get("requires", {})
    if not isinstance(requires, dict):
        return SkillRequirements()

    install_data = requires.get("install", [])
    if not isinstance(install_data, list):
        install_data = []

    return SkillRequirements(
        bins=requires.get("bins", []),
        any_bins=requires.get("any_bins", []),
        env=requires.get("env", []),
        os=requires.get("os", []),
        apt_packages=requires.get("apt_packages", []),
        python_packages=requires.get("python_packages", []),
        python_tools=requires.get("python_tools", []),
        install=self._parse_install_specs(install_data),
    )
```

### Effort

**M** (half day) - New dataclass with parsing, but conceptually straightforward.

### Priority

**Medium** - Helps users self-serve when skills are unavailable. Nice UX improvement.

---

## Gap 5: Config-Based Enable/Disable

### What Ash is Missing

Clawdbot allows disabling specific skills via config:
```toml
[skills.entries.web-browser]
enabled = false
```

Ash has no equivalent. The only way to disable a skill is to delete it or fail its requirements check.

### Reference

**Best implementation:** clawdbot (`skills.ts` lines 331-341)
```typescript
function shouldIncludeSkill(params: {
  entry: SkillEntry;
  config?: ClawdbotConfig;
}): boolean {
  const { entry, config } = params;
  const skillKey = resolveSkillKey(entry.skill, entry);
  const skillConfig = resolveSkillConfig(config, skillKey);

  if (skillConfig?.enabled === false) return false;
  // ... rest of checks
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/config/models.py`
- `/home/dcramer/src/ash/src/ash/skills/registry.py`

### Proposed Changes

```python
# In config/models.py:

class SkillEntryConfig(BaseModel):
    """Configuration for a specific skill."""

    enabled: bool = True  # Set to false to disable


class SkillsConfig(BaseModel):
    """Configuration for skills system."""

    enable_user_skills: bool = True
    enable_workspace_skills: bool = True
    extra_dirs: list[str] = Field(default_factory=list)
    ignored_skills: list[str] = Field(default_factory=list)
    include_skills: list[str] = Field(default_factory=list)

    # Per-skill configuration: [skills.entries.<skill-name>]
    entries: dict[str, SkillEntryConfig] = Field(default_factory=dict)
```

```python
# In registry.py:

from ash.config.models import SkillsConfig

class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}
        self._skill_sources: dict[str, Path] = {}
        self._ignored_patterns: list[str] = []
        self._include_patterns: list[str] = []
        self._skill_config: dict[str, bool] = {}  # skill_name -> enabled

    def set_skill_config(self, entries: dict[str, Any]) -> None:
        """Set per-skill configuration.

        Args:
            entries: Dict of skill_name -> SkillEntryConfig or dict with 'enabled'.
        """
        self._skill_config = {}
        for name, config in entries.items():
            if hasattr(config, 'enabled'):
                self._skill_config[name] = config.enabled
            elif isinstance(config, dict):
                self._skill_config[name] = config.get('enabled', True)

    def _should_include_skill(self, name: str) -> bool:
        """Check if a skill should be included based on filters and config."""
        # Check explicit enable/disable first
        if name in self._skill_config and not self._skill_config[name]:
            logger.debug(f"Skill '{name}' disabled in config")
            return False

        # Ignore patterns take precedence over include patterns
        for pattern in self._ignored_patterns:
            if fnmatch.fnmatch(name, pattern):
                logger.debug(f"Skill '{name}' excluded by ignore pattern '{pattern}'")
                return False

        # If no include patterns, include everything not ignored
        if not self._include_patterns:
            return True

        # Must match at least one include pattern
        for pattern in self._include_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True

        logger.debug(f"Skill '{name}' excluded (no include pattern matched)")
        return False
```

Example config usage:
```toml
[skills.entries.web-browser]
enabled = false

[skills.entries.expensive-api-skill]
enabled = false
```

### Effort

**S** (1-2 hours) - Simple boolean check with minimal config changes.

### Priority

**Medium** - Common need when testing or when skills conflict.

---

## Gap 6: API Key Injection

### What Ash is Missing

Clawdbot has a `primaryEnv` + `apiKey` pattern that allows skills to declare which env var they need, and config to provide the API key:

Skill declares:
```yaml
metadata: {"clawdbot": {"primaryEnv": "OPENAI_API_KEY", ...}}
```

Config provides:
```toml
[skills.entries.gpt-skill]
apiKey = "sk-..."
```

At runtime, clawdbot injects the API key into the environment before running the skill.

Ash has no way to inject API keys per-skill. Users must set env vars globally.

### Reference

**Best implementation:** clawdbot (`skills.ts` lines 410-443)
```typescript
export function applySkillEnvOverrides(params: {
  skills: SkillEntry[];
  config?: ClawdbotConfig;
}) {
  const { skills, config } = params;
  const updates: Array<{ key: string; prev: string | undefined }> = [];

  for (const entry of skills) {
    const skillKey = resolveSkillKey(entry.skill, entry);
    const skillConfig = resolveSkillConfig(config, skillKey);
    if (!skillConfig) continue;

    // Inject custom env vars
    if (skillConfig.env) {
      for (const [envKey, envValue] of Object.entries(skillConfig.env)) {
        if (!envValue || process.env[envKey]) continue;
        updates.push({ key: envKey, prev: process.env[envKey] });
        process.env[envKey] = envValue;
      }
    }

    // Inject apiKey as primaryEnv
    const primaryEnv = entry.clawdbot?.primaryEnv;
    if (primaryEnv && skillConfig.apiKey && !process.env[primaryEnv]) {
      updates.push({ key: primaryEnv, prev: process.env[primaryEnv] });
      process.env[primaryEnv] = skillConfig.apiKey;
    }
  }

  // Return cleanup function
  return () => {
    for (const update of updates) {
      if (update.prev === undefined) delete process.env[update.key];
      else process.env[update.key] = update.prev;
    }
  };
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/config/models.py`
- `/home/dcramer/src/ash/src/ash/skills/base.py`
- `/home/dcramer/src/ash/src/ash/skills/registry.py`

### Proposed Changes

```python
# In config/models.py:

class SkillEntryConfig(BaseModel):
    """Configuration for a specific skill."""

    enabled: bool = True

    # API key for this skill (injected as primary_env)
    api_key: SecretStr | None = None

    # Additional environment variables for this skill
    env: dict[str, str] = Field(default_factory=dict)
```

```python
# In base.py, add to SkillDefinition:

@dataclass
class SkillDefinition:
    """Skill definition - loaded from SKILL.md files."""

    name: str
    description: str
    instructions: str
    allowed_tools: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    requires: SkillRequirements = field(default_factory=SkillRequirements)
    skill_path: Path | None = None

    # Primary environment variable for API key injection
    # If set, config can provide api_key which gets injected as this env var
    primary_env: str | None = None
```

```python
# In registry.py, add environment resolution:

import os
from contextlib import contextmanager
from pydantic import SecretStr

class SkillRegistry:
    # ... existing code ...

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}
        self._skill_sources: dict[str, Path] = {}
        self._ignored_patterns: list[str] = []
        self._include_patterns: list[str] = []
        self._skill_config: dict[str, Any] = {}  # Full config per skill

    def set_skill_config(self, entries: dict[str, Any]) -> None:
        """Set per-skill configuration."""
        self._skill_config = entries

    @contextmanager
    def skill_environment(self, skill_name: str):
        """Context manager that injects skill-specific environment variables.

        Usage:
            with registry.skill_environment("my-skill"):
                # API keys and env vars are available here
                run_skill()
        """
        skill = self._skills.get(skill_name)
        config = self._skill_config.get(skill_name)

        if not skill or not config:
            yield
            return

        updates: list[tuple[str, str | None]] = []  # (key, original_value)

        try:
            # Inject custom env vars
            env_vars = getattr(config, 'env', {}) or {}
            for key, value in env_vars.items():
                if value and not os.environ.get(key):
                    updates.append((key, os.environ.get(key)))
                    os.environ[key] = value

            # Inject API key as primary_env
            api_key = getattr(config, 'api_key', None)
            primary_env = skill.primary_env
            if primary_env and api_key and not os.environ.get(primary_env):
                updates.append((primary_env, os.environ.get(primary_env)))
                # Handle SecretStr
                key_value = api_key.get_secret_value() if hasattr(api_key, 'get_secret_value') else str(api_key)
                os.environ[primary_env] = key_value

            yield
        finally:
            # Restore original environment
            for key, original in updates:
                if original is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original


    def _load_markdown_skill(self, path: Path, default_name: str | None = None) -> None:
        """Load a skill from a markdown file with YAML frontmatter."""
        # ... existing parsing code ...

        # Extract primary_env from frontmatter
        primary_env = data.get("primary_env")

        skill = self._create_skill_definition(
            name=name,
            description=data["description"],
            instructions=instructions,
            data=data,
            skill_path=skill_path,
        )
        # Set primary_env if present
        if primary_env:
            skill.primary_env = primary_env

        self._register_skill(skill, path)
```

### Effort

**M** (half day) - Requires context manager pattern and config wiring.

### Priority

**High** - Essential for skills that need API keys (search, external services).

---

## Gap 7: Skill Validation Warnings

### What Ash is Missing

Pi-mono validates skill names thoroughly and emits warnings:
- Name must match parent directory name
- Name must be lowercase a-z, 0-9, hyphens only
- Name cannot start/end with hyphen
- Name cannot have consecutive hyphens
- Max 64 characters
- Description max 1024 characters
- Unknown frontmatter fields generate warnings

Ash has minimal validation - just checks for required fields.

### Reference

**Best implementation:** pi-mono (`skills.ts` lines 99-151)
```typescript
function validateName(name: string, parentDirName: string): string[] {
  const errors: string[] = [];

  if (name !== parentDirName) {
    errors.push(`name "${name}" does not match parent directory "${parentDirName}"`);
  }

  if (name.length > MAX_NAME_LENGTH) {
    errors.push(`name exceeds ${MAX_NAME_LENGTH} characters (${name.length})`);
  }

  if (!/^[a-z0-9-]+$/.test(name)) {
    errors.push(`name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)`);
  }

  if (name.startsWith("-") || name.endsWith("-")) {
    errors.push(`name must not start or end with a hyphen`);
  }

  if (name.includes("--")) {
    errors.push(`name must not contain consecutive hyphens`);
  }

  return errors;
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/skills/registry.py`

### Proposed Changes

```python
# In registry.py, add validation:

import re

# Validation constants
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
VALID_NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")
ALLOWED_FRONTMATTER_FIELDS = {
    "name", "description", "requires", "allowed_tools",
    "input_schema", "primary_env", "license", "compatibility", "metadata"
}


@dataclass
class SkillWarning:
    """Warning about a skill that was loaded but has issues."""
    skill_path: Path
    message: str


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}
        self._skill_sources: dict[str, Path] = {}
        self._warnings: list[SkillWarning] = []
        # ... rest of init ...

    def get_warnings(self) -> list[SkillWarning]:
        """Get validation warnings from skill loading."""
        return list(self._warnings)

    def _validate_name(self, name: str, parent_dir_name: str | None) -> list[str]:
        """Validate skill name per conventions.

        Returns list of warning messages (empty if valid).
        """
        warnings = []

        # Check name matches directory
        if parent_dir_name and name != parent_dir_name:
            warnings.append(
                f"name '{name}' does not match parent directory '{parent_dir_name}'"
            )

        # Check length
        if len(name) > MAX_NAME_LENGTH:
            warnings.append(
                f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})"
            )

        # Check characters
        if not VALID_NAME_PATTERN.match(name):
            warnings.append(
                "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
            )

        # Check hyphen placement
        if name.startswith("-") or name.endswith("-"):
            warnings.append("name must not start or end with a hyphen")

        # Check consecutive hyphens
        if "--" in name:
            warnings.append("name must not contain consecutive hyphens")

        return warnings

    def _validate_description(self, description: str | None) -> list[str]:
        """Validate skill description."""
        warnings = []

        if not description or not description.strip():
            warnings.append("description is required")
        elif len(description) > MAX_DESCRIPTION_LENGTH:
            warnings.append(
                f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})"
            )

        return warnings

    def _validate_frontmatter_fields(self, keys: list[str]) -> list[str]:
        """Check for unknown frontmatter fields."""
        warnings = []
        for key in keys:
            if key not in ALLOWED_FRONTMATTER_FIELDS:
                warnings.append(f"unknown frontmatter field '{key}'")
        return warnings

    def _load_markdown_skill(self, path: Path, default_name: str | None = None) -> None:
        """Load a skill from a markdown file with YAML frontmatter."""
        content = path.read_text()

        # Parse frontmatter
        match = FRONTMATTER_PATTERN.match(content)
        if not match:
            raise ValueError("No YAML frontmatter found (must start with ---)")

        frontmatter_yaml = match.group(1)
        instructions = content[match.end():].strip()

        data = yaml.safe_load(frontmatter_yaml)
        if not isinstance(data, dict):
            raise ValueError("Frontmatter must be a YAML mapping")

        # Name priority: frontmatter > default_name > filename stem
        name = data.get("name") or default_name or path.stem

        # Determine parent directory name for validation
        parent_dir_name = path.parent.name if path.name == "SKILL.md" else None

        # Collect all validation warnings
        all_keys = list(data.keys())
        name_warnings = self._validate_name(name, parent_dir_name)
        desc_warnings = self._validate_description(data.get("description"))
        field_warnings = self._validate_frontmatter_fields(all_keys)

        for msg in name_warnings + desc_warnings + field_warnings:
            self._warnings.append(SkillWarning(skill_path=path, message=msg))
            logger.warning(f"Skill '{name}' ({path}): {msg}")

        # Still require description to actually load
        if "description" not in data:
            raise ValueError("Skill missing required field: description")

        if not instructions:
            raise ValueError("Skill missing instructions (markdown body)")

        # ... rest of loading logic ...
```

### Effort

**S** (1-2 hours) - Straightforward validation with regex and string checks.

### Priority

**Low** - Nice for consistency but skills still work without it. Useful for `ash skill validate`.

---

## Gap 8: Symlink Deduplication

### What Ash is Missing

Pi-mono tracks realpath of skill files to avoid loading the same skill twice via symlinks. This prevents:
- Duplicate skills appearing in the list
- Potential conflicts from the same skill loaded from different paths

Current ash code doesn't handle symlinks specially - it could load the same skill twice.

### Reference

**Best implementation:** pi-mono (`skills.ts` lines 386-412)
```typescript
const realPathSet = new Set<string>();

function addSkills(result: LoadSkillsResult) {
  for (const skill of result.skills) {
    // Resolve symlinks to detect duplicate files
    let realPath: string;
    try {
      realPath = realpathSync(skill.filePath);
    } catch {
      realPath = skill.filePath;
    }

    // Skip silently if we've already loaded this exact file (via symlink)
    if (realPathSet.has(realPath)) {
      continue;
    }

    const existing = skillMap.get(skill.name);
    if (existing) {
      collisionWarnings.push({
        skillPath: skill.filePath,
        message: `name collision: "${skill.name}" already loaded from ${existing.filePath}, skipping`,
      });
    } else {
      skillMap.set(skill.name, skill);
      realPathSet.add(realPath);
    }
  }
}
```

### Files to Modify

- `/home/dcramer/src/ash/src/ash/skills/registry.py`

### Proposed Changes

```python
# In registry.py:

class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}
        self._skill_sources: dict[str, Path] = {}
        self._loaded_realpaths: set[Path] = set()  # Track resolved paths
        self._warnings: list[SkillWarning] = []
        # ... rest of init ...

    def _register_skill(self, skill: SkillDefinition, source_path: Path) -> None:
        """Register a skill, handling symlinks and overrides."""
        # Apply filters before registration
        if not self._should_include_skill(skill.name):
            return

        # Resolve symlinks to detect duplicates
        try:
            real_path = source_path.resolve()
        except OSError:
            real_path = source_path

        # Skip if we've already loaded this exact file (via symlink)
        if real_path in self._loaded_realpaths:
            logger.debug(
                f"Skipping duplicate skill file via symlink: {source_path} -> {real_path}"
            )
            return

        # Handle name collisions (different files, same name)
        if skill.name in self._skills:
            existing_source = self._skill_sources.get(skill.name)
            if existing_source and existing_source != source_path:
                logger.warning(f"Skill '{skill.name}' overwritten by {source_path}")

        # Check availability and log if not available
        is_available, reason = skill.is_available()
        if not is_available:
            logger.debug(f"Skill '{skill.name}' not available: {reason}")

        self._skills[skill.name] = skill
        self._skill_sources[skill.name] = source_path
        self._loaded_realpaths.add(real_path)
        logger.debug(f"Loaded skill: {skill.name} from {source_path}")

    def clear(self) -> None:
        """Clear all loaded skills and reset state."""
        self._skills.clear()
        self._skill_sources.clear()
        self._loaded_realpaths.clear()
        self._warnings.clear()
```

### Effort

**S** (1-2 hours) - Simple path resolution with set tracking.

### Priority

**Low** - Edge case. Most users don't symlink skills. But prevents confusing bugs when they do.

---

## Summary Table

| Gap | Description | Effort | Priority | Main Benefit |
|-----|-------------|--------|----------|--------------|
| 1 | anyBins requirement | S | Medium | Support tools with multiple implementations |
| 2 | Multi-source loading | M | **High** | Global skills + workspace overrides |
| 3 | Glob-based filtering | S | Low | Fine-grained skill control |
| 4 | Install specs | M | Medium | Help users install missing deps |
| 5 | Config enable/disable | S | Medium | Easy skill toggling |
| 6 | API key injection | M | **High** | Skills needing external APIs |
| 7 | Validation warnings | S | Low | Consistent skill naming |
| 8 | Symlink deduplication | S | Low | Prevent duplicate loading |

## Recommended Implementation Order

1. **Gap 2: Multi-source loading** (High priority, enables global skills)
2. **Gap 6: API key injection** (High priority, unblocks external service skills)
3. **Gap 5: Config enable/disable** (Medium, quick win while touching config)
4. **Gap 1: anyBins requirement** (Medium, simple addition)
5. **Gap 4: Install specs** (Medium, good UX)
6. **Gap 7: Validation warnings** (Low, nice polish)
7. **Gap 3: Glob filtering** (Low, power user feature)
8. **Gap 8: Symlink deduplication** (Low, edge case fix)
