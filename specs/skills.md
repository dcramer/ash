# Skills

> Reusable behaviors that orchestrate tools with model preferences

Files: src/ash/skills/base.py, src/ash/skills/registry.py, src/ash/skills/executor.py, src/ash/skills/bundled/, src/ash/tools/builtin/skills.py

## Requirements

### MUST

- Load bundled skills from `src/ash/skills/bundled/`
- Load workspace skills from `workspace/skills/` (can override bundled)
- Support directory format: `skills/<name>/SKILL.md` (preferred)
- Support flat markdown: `skills/<name>.md` (convenience)
- Support pure YAML: `skills/<name>.yaml` (backward compatibility)
- Each skill defines: name, description, instructions, preferred_model, required_tools
- Support skill requirements: bins, env, os filtering
- Support skill config: list of env var names with optional defaults
- Load config values from layered sources (skill config.toml → central config → env vars → defaults)
- Mark skill unavailable if required config missing
- Pass resolved config to sandbox as `SKILL_*` environment variables
- Filter unavailable skills from system prompt and iteration
- SkillRegistry discovers and loads skills from bundled + workspace
- SkillExecutor creates sub-agent loop with skill instructions as system prompt
- Validate skill availability before execution
- List skills in system prompt (via SystemPromptBuilder)
- Expose `use_skill` tool for invoking skills
- Skills can reference model aliases (e.g., "fast", "default")
- Validate required_tools exist before skill execution
- Pass skill results back to parent agent

### SHOULD

- Support skill parameters via input_schema (JSON Schema)
- Allow skills to specify max_iterations independently
- Log skill execution with duration and iteration count
- Provide clear error when referenced model alias not found
- Default skill name to filename stem if not specified
- Bundle useful starter skills (manage-skill, research, code-review, debug)

### MAY

- Support skill chaining (one skill invoking another via use_skill)
- Watch workspace/skills/ for changes and reload
- Track skill usage statistics

## Interface

### Directory Skill Format (Preferred)

```
workspace/skills/
  summarize/
    SKILL.md
    config.toml           # Optional: skill-local config values
    scripts/              # Optional: executable scripts
      main.py
  explain/
    SKILL.md
```

```markdown
<!-- workspace/skills/summarize/SKILL.md -->
---
description: Summarize text or documents concisely
preferred_model: fast
required_tools:
  - bash
max_iterations: 3
requires:
  bins:
    - pandoc
  env: []
  os:
    - linux
    - darwin
config:
  - API_KEY                   # Required (no default)
  - MAX_LENGTH=1000           # Optional with default
input_schema:
  type: object
  properties:
    content:
      type: string
      description: Text or file path to summarize
    format:
      type: string
      enum: [bullets, paragraph, tldr]
      default: bullets
  required:
    - content
---

You are a summarization assistant. Create clear, concise summaries.

Extract key points only. Maintain factual accuracy.
Use the requested format for output.
```

Note: `name` defaults to the directory name (e.g., `skills/summarize/` → `summarize`).

### Skill Config Format

Skills declare config requirements in SKILL.md. Values are provided via layered sources.

**SKILL.md config declaration:**
```yaml
config:
  - API_KEY                   # Required (no default)
  - DEFAULT_VALUE=fallback    # Optional with default
```

**Skill-local config.toml (gitignored):**
```toml
# workspace/skills/<name>/config.toml
API_KEY = "$MY_API_KEY"       # Reference env var
DEFAULT_VALUE = "custom"       # Literal value
```

**Central config.toml:**
```toml
# ~/.ash/config.toml
[skills.summarize]
API_KEY = "abc123"
```

**Resolution order (first match wins):**
1. Skill's `config.toml`
2. Central `[skills.<name>]` section
3. Environment variable by name
4. Default from SKILL.md (after `=`)

**Passed to sandbox as:**
- `SKILL_API_KEY`
- `SKILL_DEFAULT_VALUE`

### YAML Skill Format (Backward Compatibility)

```yaml
# workspace/skills/summarize.yaml
name: summarize
description: Summarize text or documents concisely
preferred_model: fast
required_tools:
  - bash
max_iterations: 3
input_schema:
  type: object
  properties:
    content:
      type: string
  required:
    - content
instructions: |
  You are a summarization assistant.
```

### Python Classes

```python
@dataclass
class SkillRequirements:
    """Requirements for a skill to be available."""
    bins: list[str] = field(default_factory=list)  # Required binaries in PATH
    env: list[str] = field(default_factory=list)   # Required environment variables
    os: list[str] = field(default_factory=list)    # Supported OS (darwin, linux, windows)

    def check(self) -> tuple[bool, str | None]:
        """Check if requirements are met. Returns (is_met, error_message)."""
        ...

@dataclass
class SkillDefinition:
    """Skill loaded from workspace."""
    name: str
    description: str
    instructions: str
    preferred_model: str | None = None
    required_tools: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 5
    requires: SkillRequirements = field(default_factory=SkillRequirements)
    config: list[str] = field(default_factory=list)  # Env var names with optional =default
    config_values: dict[str, str] = field(default_factory=dict)  # Resolved values
    skill_path: Path | None = None  # Path to skill directory

    def is_available(self) -> tuple[bool, str | None]:
        """Check if skill is available on current system."""
        # Check requirements first
        ok, msg = self.requires.check()
        if not ok:
            return ok, msg
        # Check config
        return self.is_config_valid()

    def is_config_valid(self) -> tuple[bool, str | None]:
        """Check if all required config values are present."""
        for item in self.config:
            name = item.split("=")[0]
            if "=" not in item and name not in self.config_values:
                return False, f"Missing required config: {name}"
        return True, None

@dataclass
class SkillContext:
    """Context passed to skill execution."""
    session_id: str | None = None
    user_id: str | None = None
    chat_id: str | None = None
    input_data: dict[str, Any] = field(default_factory=dict)

@dataclass
class SkillResult:
    """Result from skill execution."""
    content: str
    is_error: bool = False
    iterations: int = 0

    @classmethod
    def success(cls, content: str, iterations: int = 0) -> "SkillResult": ...

    @classmethod
    def error(cls, message: str) -> "SkillResult": ...
```

### Registry

```python
class SkillRegistry:
    def load_bundled(self) -> None:
        """Load bundled skills from src/ash/skills/bundled/."""
        ...

    def discover(self, workspace_path: Path, *, include_bundled: bool = True) -> None:
        """Load skills from bundled (optional) and workspace directories."""
        ...

    def get(self, name: str) -> SkillDefinition:
        """Get skill by name. Raises KeyError if not found."""
        ...

    def has(self, name: str) -> bool: ...

    def list_names(self) -> list[str]:
        """List all registered skill names (including unavailable)."""
        ...

    def list_available(self) -> list[SkillDefinition]:
        """List skills available on current system."""
        ...

    def get_definitions(self, include_unavailable: bool = False) -> list[dict[str, Any]]:
        """Get skill definitions for LLM. By default only returns available skills."""
        ...

    def __iter__(self) -> Iterator[SkillDefinition]:
        """Iterate over available skills only."""
        ...

    def __len__(self) -> int: ...
```

### Executor

```python
class SkillExecutor:
    def __init__(
        self,
        registry: SkillRegistry,
        tool_executor: ToolExecutor,
        config: AshConfig,
    ) -> None: ...

    async def execute(
        self,
        skill_name: str,
        input_data: dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """Execute skill with sub-agent loop."""
        ...
```

### LLM Tool

```python
class UseSkillTool(Tool):
    """Invoke a skill by name."""
    name = "use_skill"
    input_schema = {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "description": "Skill name"},
            "input": {"type": "object", "description": "Skill input parameters"},
        },
        "required": ["skill"],
    }
```

### Bundled Skills

Skills shipped with Ash in `src/ash/skills/bundled/`:

| Skill | Description |
|-------|-------------|
| manage-skill | Create, edit, or view skills in the workspace |
| research | Research a topic using web search and memory |
| code-review | Review code for bugs, security issues, and improvements |
| debug | Systematically debug issues in code or systems |

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|
| Registry.discover() called | Bundled + workspace skills loaded | Bundled first, workspace can override |
| Skills discovered | Listed in system prompt | Via SystemPromptBuilder |
| `use_skill(summarize, {content: "..."})` | SkillResult with summary | Sub-agent executes |
| Skill with `preferred_model: fast` | Uses `models.fast` config | Model alias resolved |
| Skill with unknown model alias | Falls back to default model | Warning logged |
| Skill requires unavailable tool | Error before execution | Validation fails |
| Skill exceeds max_iterations | Returns partial result | With limit message |
| Empty workspace/skills/ | Bundled skills only | No error |
| Skill without `name` in frontmatter | Uses filename stem | e.g., `foo.md` → `foo` |
| Skill with `requires.bins` not in PATH | Filtered from prompt/iteration | Still registered |
| Skill with `requires.env` not set | Filtered from prompt/iteration | Still registered |
| Skill with `requires.os` not matching | Filtered from prompt/iteration | Still registered |
| Workspace skill same name as bundled | Workspace overrides bundled | Customization |
| Skill with `config` declared | Registry loads config.toml if exists | Layered resolution |
| Config value `$VAR` | Resolved from environment | Env var expansion |
| Required config missing | Skill marked unavailable | Filtered from prompt |
| Config provided | Passed as `SKILL_*` env vars to sandbox | Uppercase, prefixed |

## Errors

| Condition | Response |
|-----------|----------|
| Skill not found | SkillResult.error("Skill 'name' not found") |
| Skill not available | SkillResult.error("Skill 'name' not available: <reason>") |
| Required tool unavailable | SkillResult.error("Skill requires tool 'bash' which is not available") |
| Invalid input schema | SkillResult.error("Invalid input: <validation error>") |
| Missing frontmatter | Logged warning, skill skipped during discovery |
| Missing description | Logged warning, skill skipped |
| Empty instructions | Logged warning, skill skipped |
| Model alias not found | Uses default model, logs warning |
| Required config missing | SkillResult.error("Skill 'name' not available: Missing required config: X") |
| Config.toml parse error | Logged warning, config values empty |

## Verification

```bash
uv run pytest tests/test_skills.py -v

# Test bundled skills loaded
uv run ash chat "What skills are available?"
# Should show: manage-skill, research, code-review, debug

# Test skill creation via manage-skill
uv run ash chat "Use the manage-skill skill to create a greeting skill"

# Test skill with requirements
mkdir -p workspace/skills/darwin-only
cat > workspace/skills/darwin-only/SKILL.md << 'EOF'
---
description: macOS-only skill
requires:
  os:
    - darwin
---

This skill only works on macOS.
EOF

# Verify filtering (skill should not appear on Linux)
uv run ash chat "What skills are available?"

# Test skill config
mkdir -p workspace/skills/config-test
cat > workspace/skills/config-test/SKILL.md << 'EOF'
---
description: Test config loading
required_tools:
  - bash
config:
  - TEST_KEY
  - OPTIONAL_KEY=default
---
Echo the SKILL_* env vars.
EOF

# Without config, skill should be unavailable
uv run ash chat "What skills are available?"
# config-test should NOT be listed

# Add config
cat > workspace/skills/config-test/config.toml << 'EOF'
TEST_KEY = "hello"
EOF

# Now skill should be available
uv run ash chat "use the config-test skill"
# Should see SKILL_TEST_KEY=hello, SKILL_OPTIONAL_KEY=default
```

- Bundled skills loaded from src/ash/skills/bundled/
- Workspace skills loaded from workspace/skills/
- Workspace skills can override bundled skills
- Skills listed in system prompt
- Directory format `<name>/SKILL.md` loads correctly
- Flat markdown files still supported
- YAML files still supported
- use_skill executes skill with sub-agent
- Model alias resolution works
- Missing tools detected before execution
- Invalid files skipped with warning
- Skills with unmet requirements filtered from prompt
- Skills with unmet requirements return error on execution
- Skills with `config` load values from config.toml
- Config values resolved from layered sources
- Required config missing marks skill unavailable
- Config passed as SKILL_* env vars to sandbox
