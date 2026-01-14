# Skills System Comparison

A comprehensive comparison of skills systems across four agent codebases: ash (Python), archer (TypeScript), clawdbot (TypeScript), and pi-mono (TypeScript).

## Overview

Skills are reusable instruction sets that extend agent capabilities without modifying core code. They provide domain-specific knowledge, tool usage patterns, and workflow guidance that gets injected into the system prompt. Each codebase implements skills differently based on their distribution model (bundled vs user-managed), requirements checking sophistication, and state management needs.

| Aspect | ash | archer | clawdbot | pi-mono |
|--------|-----|--------|----------|---------|
| Language | Python | TypeScript | TypeScript | TypeScript |
| Skill Format | SKILL.md with YAML frontmatter | SKILL.md with YAML frontmatter | SKILL.md with YAML frontmatter | SKILL.md with YAML frontmatter |
| Bundled Skills | None | None | 49 skills | None |
| Requirements Check | bins, env, os | None | bins, anyBins, env, config, os | ignoredSkills, includeSkills (glob patterns) |
| Install Automation | None | None | brew, npm, go, uv | None |
| State Management | JSON files per skill | None | None | None |
| Loading Sources | workspace/skills/ | workspace/skills/, channel/skills/ | extra < bundled < managed < workspace | codex-user, claude-user, claude-project, pi-user, pi-project, custom |
| Platform Filtering | Yes (os field) | No | Yes (os field) | No |
| Config-Based Enable/Disable | No | No | Yes | No |

## Detailed Analysis

### 1. ash (Python)

**Location:** `/home/dcramer/src/ash/src/ash/skills/`

ash implements a workspace-only skills system with requirements checking and persistent state storage. Skills are loaded from `workspace/skills/` only (no bundled skills).

#### Skill Definition

```python
# src/ash/skills/base.py
@dataclass
class SkillRequirements:
    """Requirements for a skill to be available."""
    bins: list[str] = field(default_factory=list)   # Required binaries (all must exist)
    env: list[str] = field(default_factory=list)    # Required env vars (all must be set)
    os: list[str] = field(default_factory=list)     # Supported OS: darwin, linux, windows

    def check(self) -> tuple[bool, str | None]:
        # Check OS
        if self.os:
            current_os = platform.system().lower()
            if current_os not in self.os:
                return False, f"Requires OS: {', '.join(self.os)} (current: {current_os})"

        # Check binaries
        for bin_name in self.bins:
            if not shutil.which(bin_name):
                return False, f"Requires binary: {bin_name}"

        # Check environment variables
        for env_var in self.env:
            if not os.environ.get(env_var):
                return False, f"Requires environment variable: {env_var}"

        return True, None


@dataclass
class SkillDefinition:
    """Skill definition - loaded from SKILL.md files."""
    name: str
    description: str
    instructions: str
    allowed_tools: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    requires: SkillRequirements = field(default_factory=SkillRequirements)
    skill_path: Path | None = None  # For {baseDir} substitution
```

#### Registry with Discovery

```python
# src/ash/skills/registry.py
class SkillRegistry:
    def discover(self, workspace_path: Path, *, include_bundled: bool = True) -> None:
        """Load skills from workspace directory.

        Supports:
        - Directory format: skills/<name>/SKILL.md (preferred)
        - Flat markdown: skills/<name>.md (convenience)
        - Pure YAML: skills/<name>.yaml (backward compatibility)
        """
        skills_dir = workspace_path / "skills"
        if not skills_dir.exists():
            return
        self._load_from_directory(skills_dir, source="workspace")

    def list_available(self) -> list[SkillDefinition]:
        """List skills available on the current system."""
        return [skill for skill in self._skills.values() if skill.is_available()[0]]
```

#### State Management

ash uniquely provides persistent state storage for skills:

```python
# src/ash/skills/state.py
class SkillStateStore:
    """File-based state storage for skills.

    Storage format (JSON):
    {
        "global": { "key1": "value1" },
        "users": { "user-123": { "key1": "user-specific-value" } }
    }
    """

    def get(self, skill_name: str, key: str, user_id: str | None = None) -> Any | None:
        """Get a skill state value (global or per-user)."""

    def set(self, skill_name: str, key: str, value: Any, user_id: str | None = None) -> None:
        """Set a skill state value (JSON-serializable)."""

    def delete(self, skill_name: str, key: str, user_id: str | None = None) -> bool:
        """Delete a skill state value."""
```

#### Skill File Format

```markdown
---
description: What the skill does
requires:
  bins: [git, gh]
  env: [GITHUB_TOKEN]
  os: [darwin, linux]
---

Instructions go here as markdown body.
Use {baseDir} for paths relative to skill directory.
```

---

### 2. archer (TypeScript)

**Location:** `/home/dcramer/src/archer/src/agent.ts`

archer uses a minimal skills system inherited from pi-coding-agent. Skills are loaded at runtime from workspace and channel directories with no requirements checking.

#### Skill Loading

```typescript
// src/agent.ts
import { formatSkillsForPrompt, loadSkillsFromDir, type Skill } from "@mariozechner/pi-coding-agent";

function loadMomSkills(channelDir: string, workspacePath: string): Skill[] {
    const skillMap = new Map<string, Skill>();
    const hostWorkspacePath = join(channelDir, "..");

    // Helper to translate host paths to container paths
    const translatePath = (hostPath: string): string => {
        if (hostPath.startsWith(hostWorkspacePath)) {
            return workspacePath + hostPath.slice(hostWorkspacePath.length);
        }
        return hostPath;
    };

    // Load workspace-level skills (global)
    const workspaceSkillsDir = join(hostWorkspacePath, "skills");
    for (const skill of loadSkillsFromDir({ dir: workspaceSkillsDir, source: "workspace" }).skills) {
        skill.filePath = translatePath(skill.filePath);
        skill.baseDir = translatePath(skill.baseDir);
        skillMap.set(skill.name, skill);
    }

    // Load channel-specific skills (override workspace skills on collision)
    const channelSkillsDir = join(channelDir, "skills");
    for (const skill of loadSkillsFromDir({ dir: channelSkillsDir, source: "channel" }).skills) {
        skill.filePath = translatePath(skill.filePath);
        skill.baseDir = translatePath(skill.baseDir);
        skillMap.set(skill.name, skill);  // Channel overrides workspace
    }

    return Array.from(skillMap.values());
}
```

#### System Prompt Integration

```typescript
// Embedded in system prompt
const systemPrompt = `
## Skills (Custom CLI Tools)
You can create reusable CLI tools for recurring tasks.

### Creating Skills
Store in \`${workspacePath}/skills/<name>/\` (global) or \`${chatPath}/skills/<name>/\` (chat-specific).
Each skill directory needs a \`SKILL.md\` with YAML frontmatter:

\`\`\`markdown
---
name: skill-name
description: Short description of what this skill does
---

# Skill Name
Usage instructions, examples, etc.
Scripts are in: {baseDir}/
\`\`\`

### Available Skills
${skills.length > 0 ? formatSkillsForPrompt(skills) : "(no skills installed yet)"}
`;
```

#### Skill Precedence

- Workspace skills loaded first
- Channel skills override workspace skills on name collision
- No bundled skills

---

### 3. clawdbot (TypeScript)

**Location:** `/home/dcramer/src/clawdbot/src/agents/skills.ts`

clawdbot has the most sophisticated skills system with 49 bundled skills, comprehensive requirements checking, install automation specs, and config-based enable/disable.

#### Metadata Types

```typescript
// src/agents/skills.ts
export type SkillInstallSpec = {
    id?: string;
    kind: "brew" | "node" | "go" | "uv";
    label?: string;
    bins?: string[];
    formula?: string;    // For brew
    package?: string;    // For node/go
    module?: string;     // For uv
};

export type ClawdbotSkillMetadata = {
    always?: boolean;           // Always include regardless of requirements
    skillKey?: string;          // Key for config lookup
    primaryEnv?: string;        // Primary env var (for apiKey shorthand)
    emoji?: string;             // Display emoji
    homepage?: string;          // Documentation URL
    os?: string[];              // Platform filter: darwin, linux, win32
    requires?: {
        bins?: string[];        // All must exist
        anyBins?: string[];     // At least one must exist
        env?: string[];         // Required env vars
        config?: string[];      // Required config paths (dot notation)
    };
    install?: SkillInstallSpec[];  // Installation instructions
};
```

#### Requirements Checking

```typescript
function shouldIncludeSkill(params: { entry: SkillEntry; config?: ClawdbotConfig }): boolean {
    const { entry, config } = params;
    const skillKey = resolveSkillKey(entry.skill, entry);
    const skillConfig = resolveSkillConfig(config, skillKey);
    const allowBundled = normalizeAllowlist(config?.skills?.allowBundled);
    const osList = entry.clawdbot?.os ?? [];

    // Config-based disable
    if (skillConfig?.enabled === false) return false;

    // Bundled allowlist filter
    if (!isBundledSkillAllowed(entry, allowBundled)) return false;

    // Platform filter
    if (osList.length > 0 && !osList.includes(resolveRuntimePlatform())) return false;

    // Always-include flag
    if (entry.clawdbot?.always === true) return true;

    // Required binaries (all must exist)
    const requiredBins = entry.clawdbot?.requires?.bins ?? [];
    if (requiredBins.length > 0) {
        for (const bin of requiredBins) {
            if (!hasBinary(bin)) return false;
        }
    }

    // Any binaries (at least one must exist)
    const requiredAnyBins = entry.clawdbot?.requires?.anyBins ?? [];
    if (requiredAnyBins.length > 0) {
        const anyFound = requiredAnyBins.some((bin) => hasBinary(bin));
        if (!anyFound) return false;
    }

    // Required env vars (with config override support)
    const requiredEnv = entry.clawdbot?.requires?.env ?? [];
    if (requiredEnv.length > 0) {
        for (const envName of requiredEnv) {
            if (process.env[envName]) continue;
            if (skillConfig?.env?.[envName]) continue;
            if (skillConfig?.apiKey && entry.clawdbot?.primaryEnv === envName) continue;
            return false;
        }
    }

    // Required config paths
    const requiredConfig = entry.clawdbot?.requires?.config ?? [];
    if (requiredConfig.length > 0) {
        for (const configPath of requiredConfig) {
            if (!isConfigPathTruthy(config, configPath)) return false;
        }
    }

    return true;
}
```

#### Multi-Source Loading with Precedence

```typescript
function loadSkillEntries(workspaceDir: string, opts?: {
    config?: ClawdbotConfig;
    managedSkillsDir?: string;
    bundledSkillsDir?: string;
}): SkillEntry[] {
    const managedSkillsDir = opts?.managedSkillsDir ?? path.join(CONFIG_DIR, "skills");
    const workspaceSkillsDir = path.join(workspaceDir, "skills");
    const bundledSkillsDir = opts?.bundledSkillsDir ?? resolveBundledSkillsDir();
    const extraDirs = opts?.config?.skills?.load?.extraDirs ?? [];

    const bundledSkills = bundledSkillsDir
        ? loadSkills({ dir: bundledSkillsDir, source: "clawdbot-bundled" })
        : [];
    const extraSkills = extraDirs.flatMap((dir) =>
        loadSkills({ dir: resolveUserPath(dir), source: "clawdbot-extra" })
    );
    const managedSkills = loadSkills({ dir: managedSkillsDir, source: "clawdbot-managed" });
    const workspaceSkills = loadSkills({ dir: workspaceSkillsDir, source: "clawdbot-workspace" });

    const merged = new Map<string, Skill>();
    // Precedence: extra < bundled < managed < workspace
    for (const skill of extraSkills) merged.set(skill.name, skill);
    for (const skill of bundledSkills) merged.set(skill.name, skill);
    for (const skill of managedSkills) merged.set(skill.name, skill);
    for (const skill of workspaceSkills) merged.set(skill.name, skill);

    return Array.from(merged.values()).map((skill) => ({
        skill,
        frontmatter: parseFrontmatter(fs.readFileSync(skill.filePath, "utf-8")),
        clawdbot: resolveClawdbotMetadata(frontmatter),
    }));
}
```

#### Example Bundled Skills

**Brave Search** (requires binary + env var):
```yaml
---
name: brave-search
description: Web search and content extraction via Brave Search API.
homepage: https://brave.com/search/api
metadata: {"clawdbot":{"emoji":"🦁","requires":{"bins":["node"],"env":["BRAVE_API_KEY"]},"primaryEnv":"BRAVE_API_KEY"}}
---
```

**Apple Notes** (macOS only with install spec):
```yaml
---
name: apple-notes
description: Manage Apple Notes via the `memo` CLI on macOS.
homepage: https://github.com/antoniorodr/memo
metadata: {"clawdbot":{"emoji":"📝","os":["darwin"],"requires":{"bins":["memo"]},"install":[{"id":"brew","kind":"brew","formula":"antoniorodr/memo/memo","bins":["memo"],"label":"Install memo via Homebrew"}]}}
---
```

**tmux** (multi-platform with binary requirement):
```yaml
---
name: tmux
description: Remote-control tmux sessions for interactive CLIs.
metadata: {"clawdbot":{"emoji":"🧵","os":["darwin","linux"],"requires":{"bins":["tmux"]}}}
---
```

#### Config-Based API Key Injection

```typescript
export function applySkillEnvOverrides(params: { skills: SkillEntry[]; config?: ClawdbotConfig }) {
    const { skills, config } = params;

    for (const entry of skills) {
        const skillKey = resolveSkillKey(entry.skill, entry);
        const skillConfig = resolveSkillConfig(config, skillKey);
        if (!skillConfig) continue;

        // Apply env overrides from config
        if (skillConfig.env) {
            for (const [envKey, envValue] of Object.entries(skillConfig.env)) {
                if (!envValue || process.env[envKey]) continue;
                process.env[envKey] = envValue;
            }
        }

        // Apply apiKey shorthand to primaryEnv
        const primaryEnv = entry.clawdbot?.primaryEnv;
        if (primaryEnv && skillConfig.apiKey && !process.env[primaryEnv]) {
            process.env[primaryEnv] = skillConfig.apiKey;
        }
    }
}
```

---

### 4. pi-mono (TypeScript)

**Location:** `/home/dcramer/src/pi-mono/packages/coding-agent/src/core/skills.ts`

pi-mono provides the foundational skills loader used by archer and clawdbot. It focuses on multi-source discovery with collision detection and glob-based filtering.

#### Skill Interface

```typescript
// packages/coding-agent/src/core/skills.ts
export interface Skill {
    name: string;
    description: string;
    filePath: string;
    baseDir: string;
    source: string;
}

export interface SkillWarning {
    skillPath: string;
    message: string;
}

export interface LoadSkillsResult {
    skills: Skill[];
    warnings: SkillWarning[];
}
```

#### Validation per Agent Skills Spec

```typescript
const ALLOWED_FRONTMATTER_FIELDS = new Set([
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
]);

const MAX_NAME_LENGTH = 64;
const MAX_DESCRIPTION_LENGTH = 1024;

function validateName(name: string, parentDirName: string): string[] {
    const errors: string[] = [];

    if (name !== parentDirName) {
        errors.push(`name "${name}" does not match parent directory "${parentDirName}"`);
    }
    if (name.length > MAX_NAME_LENGTH) {
        errors.push(`name exceeds ${MAX_NAME_LENGTH} characters`);
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

#### Multi-Source Discovery

```typescript
export interface LoadSkillsOptions extends SkillsSettings {
    cwd?: string;
    agentDir?: string;
}

export function loadSkills(options: LoadSkillsOptions = {}): LoadSkillsResult {
    const {
        cwd = process.cwd(),
        agentDir,
        enableCodexUser = true,
        enableClaudeUser = true,
        enableClaudeProject = true,
        enablePiUser = true,
        enablePiProject = true,
        customDirectories = [],
        ignoredSkills = [],
        includeSkills = [],
    } = options;

    const skillMap = new Map<string, Skill>();
    const realPathSet = new Set<string>();  // Dedupe symlinks

    function addSkills(result: LoadSkillsResult) {
        for (const skill of result.skills) {
            // Apply ignore filter (glob patterns)
            if (matchesIgnorePatterns(skill.name)) continue;
            // Apply include filter (glob patterns)
            if (!matchesIncludePatterns(skill.name)) continue;

            // Dedupe by realpath (handle symlinks)
            const realPath = realpathSync(skill.filePath);
            if (realPathSet.has(realPath)) continue;

            // Handle collisions (first wins)
            if (skillMap.has(skill.name)) {
                warnings.push({
                    skillPath: skill.filePath,
                    message: `name collision: "${skill.name}" already loaded`,
                });
                continue;
            }

            skillMap.set(skill.name, skill);
            realPathSet.add(realPath);
        }
    }

    // Load from all enabled sources
    if (enableCodexUser) {
        addSkills(loadSkillsFromDir(join(homedir(), ".codex", "skills"), "codex-user"));
    }
    if (enableClaudeUser) {
        addSkills(loadSkillsFromDir(join(homedir(), ".claude", "skills"), "claude-user"));
    }
    if (enableClaudeProject) {
        addSkills(loadSkillsFromDir(resolve(cwd, ".claude", "skills"), "claude-project"));
    }
    if (enablePiUser) {
        addSkills(loadSkillsFromDir(join(agentDir, "skills"), "user"));
    }
    if (enablePiProject) {
        addSkills(loadSkillsFromDir(resolve(cwd, ".pi", "skills"), "project"));
    }
    for (const customDir of customDirectories) {
        addSkills(loadSkillsFromDir(customDir, "custom"));
    }

    return { skills: Array.from(skillMap.values()), warnings };
}
```

#### Prompt Formatting

```typescript
export function formatSkillsForPrompt(skills: Skill[]): string {
    if (skills.length === 0) return "";

    const lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "",
        "<available_skills>",
    ];

    for (const skill of skills) {
        lines.push("  <skill>");
        lines.push(`    <name>${escapeXml(skill.name)}</name>`);
        lines.push(`    <description>${escapeXml(skill.description)}</description>`);
        lines.push(`    <location>${escapeXml(skill.filePath)}</location>`);
        lines.push("  </skill>");
    }

    lines.push("</available_skills>");
    return lines.join("\n");
}
```

#### SDK Usage Example

```typescript
// examples/sdk/04-skills.ts
import { createAgentSession, discoverSkills, type Skill } from "@mariozechner/pi-coding-agent";

// Discover all skills from standard locations
const { skills: allSkills, warnings } = discoverSkills();

// Filter to specific skills
const filteredSkills = allSkills.filter((s) =>
    s.name.includes("browser") || s.name.includes("search")
);

// Or define custom skills inline
const customSkill: Skill = {
    name: "my-skill",
    description: "Custom project instructions",
    filePath: "/virtual/SKILL.md",
    baseDir: "/virtual",
    source: "custom",
};

// Use filtered + custom skills
await createAgentSession({
    skills: [...filteredSkills, customSkill],
});

// Discovery with glob filtering:
// discoverSkills(process.cwd(), undefined, {
//   ignoredSkills: ["browser-tools"],  // glob patterns to exclude
//   includeSkills: ["brave-*"],        // glob patterns to include
// })
```

---

## Key Differences

### Bundled vs User-Managed Skills

| Codebase | Bundled Skills | Distribution Model |
|----------|----------------|-------------------|
| ash | None | User creates all skills in workspace |
| archer | None | Inherits from pi-coding-agent, user-managed |
| clawdbot | 49 skills | Rich ecosystem of pre-built integrations |
| pi-mono | None | Foundation library, no bundled content |

### Requirements Checking Sophistication

| Codebase | Binaries | Env Vars | Platform | Config | Install Specs |
|----------|----------|----------|----------|--------|---------------|
| ash | All must exist | All must be set | darwin/linux/windows | No | No |
| archer | No checking | No checking | No checking | No | No |
| clawdbot | All or any | With config override | darwin/linux/win32 | Dot-path config values | brew/npm/go/uv |
| pi-mono | No checking | No checking | No checking | No | No |

### Skill Loading Sources

| Codebase | Sources | Precedence |
|----------|---------|------------|
| ash | workspace/skills/ only | Single source |
| archer | workspace/skills/, channel/skills/ | Channel overrides workspace |
| clawdbot | extra, bundled, managed (~/.clawdbot/skills/), workspace | Later sources override earlier |
| pi-mono | codex-user, claude-user, claude-project, pi-user, pi-project, custom | First loaded wins (with collision warnings) |

### State Management

| Codebase | Persistent State | Scope |
|----------|-----------------|-------|
| ash | JSON files per skill | Global and per-user |
| archer | None | N/A |
| clawdbot | None | N/A |
| pi-mono | None | N/A |

### Special Features

| Feature | ash | archer | clawdbot | pi-mono |
|---------|-----|--------|----------|---------|
| `{baseDir}` substitution | Yes | Yes | Yes | Yes |
| Config-based enable/disable | No | No | Yes | No |
| API key injection | No | No | Yes (primaryEnv + apiKey) | No |
| Glob filtering | No | No | No | Yes (ignoredSkills/includeSkills) |
| Symlink deduplication | No | No | No | Yes |
| Skill validation warnings | Basic | No | No | Comprehensive (name format, description length) |

---

## Recommendations for ash

Based on this comparison, here are potential improvements ash could adopt:

### 1. Add anyBins Requirement Check

clawdbot's `anyBins` allows specifying alternative binaries where at least one must exist:

```python
@dataclass
class SkillRequirements:
    bins: list[str] = field(default_factory=list)       # All must exist
    any_bins: list[str] = field(default_factory=list)   # At least one must exist
    env: list[str] = field(default_factory=list)
    os: list[str] = field(default_factory=list)

    def check(self) -> tuple[bool, str | None]:
        # ... existing checks ...

        # Check any_bins (at least one must exist)
        if self.any_bins:
            found = any(shutil.which(bin_name) for bin_name in self.any_bins)
            if not found:
                return False, f"Requires at least one of: {', '.join(self.any_bins)}"

        return True, None
```

### 2. Consider Bundled Skills

clawdbot's 49 bundled skills demonstrate the value of providing a curated set of integrations. ash could consider bundling common skills for:
- Git/GitHub workflows
- Common shell patterns
- Documentation generation

However, this conflicts with ash's "simplicity wins" principle. An alternative is a separate skills repository that users can selectively install.

### 3. Install Spec Documentation

clawdbot's install specs provide actionable installation instructions. ash could add an optional `install` field without automation:

```yaml
---
description: Web search skill
requires:
  bins: [brave-search]
  env: [BRAVE_API_KEY]
install:
  - label: "Install via pip"
    command: "pip install brave-search-cli"
---
```

### 4. Glob-Based Filtering

pi-mono's `ignoredSkills` and `includeSkills` glob patterns provide flexible filtering:

```python
class SkillRegistry:
    def discover(
        self,
        workspace_path: Path,
        ignored_patterns: list[str] | None = None,
        include_patterns: list[str] | None = None,
    ) -> None:
        """Load skills with optional glob filtering."""
```

### 5. Multi-Source Loading

pi-mono supports loading from multiple standard locations (~/.claude/skills/, .claude/skills/, etc.). ash could add similar support:

```python
def discover_all(
    self,
    workspace_path: Path,
    user_skills_enabled: bool = True,
) -> None:
    """Load skills from workspace and optionally user directories."""
    if user_skills_enabled:
        user_skills_dir = Path.home() / ".ash" / "skills"
        if user_skills_dir.exists():
            self._load_from_directory(user_skills_dir, source="user")

    # Workspace overrides user
    self._load_from_directory(workspace_path / "skills", source="workspace")
```

### 6. Skill Validation Warnings

pi-mono validates skill names against a spec (lowercase, no consecutive hyphens, etc.) and reports warnings. ash could add similar validation:

```python
def _validate_skill(self, name: str, path: Path) -> list[str]:
    """Validate skill against naming conventions."""
    warnings = []
    if not re.match(r'^[a-z0-9-]+$', name):
        warnings.append(f"Name should be lowercase with hyphens only: {name}")
    if name != path.parent.name:
        warnings.append(f"Name '{name}' doesn't match directory '{path.parent.name}'")
    return warnings
```

---

## Summary

Each codebase has evolved its skills system based on specific requirements:

- **ash**: Workspace-only with requirements checking and unique state management, aligned with "simplicity wins" principle
- **archer**: Minimal implementation inheriting from pi-coding-agent, channel-specific override capability
- **clawdbot**: Most sophisticated with 49 bundled skills, comprehensive requirements, install automation, and config integration
- **pi-mono**: Foundation library with multi-source discovery, validation, and glob filtering

The key architectural differences are:

1. **Distribution Model**: clawdbot bundles skills; others are user-managed
2. **Requirements Complexity**: clawdbot has bins/anyBins/env/config/os; ash has bins/env/os; others have none
3. **State**: Only ash provides persistent skill state storage
4. **Loading Sources**: pi-mono supports the most sources; ash supports only workspace
5. **Filtering**: pi-mono has glob patterns; clawdbot has config-based enable/disable; ash filters by requirements only

The SKILL.md format with YAML frontmatter is universal across all codebases, making skills portable between systems (assuming requirements are met).
