# Skills

> User-defined subagents invoked via the `use_skill` tool

Files: src/ash/skills/base.py, src/ash/skills/registry.py, src/ash/tools/builtin/skills.py

## Overview

Skills are markdown files that define specialized subagents. Unlike the current model where the main agent reads skill files, skills are now **invoked explicitly** via the `use_skill` tool and run in **isolated LLM loops** with scoped environments.

This enables:
- **API key isolation**: Skills declare needed env vars, config provides values
- **Tool restrictions**: Skills can limit which tools the subagent uses
- **Context compression**: Main agent passes relevant context, not full history
- **Model flexibility**: Skills can specify different models (e.g., haiku for simple tasks)

## Requirements

### MUST

- Load workspace skills from `workspace/skills/`
- Support directory format: `skills/<name>/SKILL.md` (preferred)
- Support flat markdown: `skills/<name>.md` (convenience)
- Support YAML format: `skills/<name>.yaml` or `.yml` (backward compatibility)
- Each skill defines: name, description, instructions
- Invoke skills via `use_skill` tool (not by reading files)
- Run skill as subagent with isolated session
- Inject env vars from config into skill execution
- Support `allowed_tools` to restrict subagent's tools
- Support `model` override per skill
- Support `max_iterations` limit per skill
- Provide CLI commands for skill management

### SHOULD

- List available skills in system prompt (name + description only)
- Log skill invocations with iteration count
- Support `enabled` flag in config to disable skills

### MAY

- Track skill usage statistics
- Support skill chaining (one skill invoking another)

## Interface

### Skill Definition Format

```
workspace/skills/
  research/
    SKILL.md
  code-review/
    SKILL.md
```

```yaml
# workspace/skills/research/SKILL.md
---
description: Research topics using Perplexity AI
authors:                       # Who created/maintains this skill
  - alice
  - bob
rationale: Enable deep research without main agent context bloat
env:                           # Env vars to inject from config
  - PERPLEXITY_API_KEY
packages:                      # System packages to install (apt)
  - jq
  - curl
allowed_tools:                 # Tool whitelist (empty = all tools)
  - bash
  - web_search
  - web_fetch
model: haiku                   # Optional model override
max_iterations: 10             # Iteration limit (default: 10)
---

You are a research assistant with access to Perplexity AI.

Given a research query, search for accurate, up-to-date information
and return a structured summary with sources.

Use the PERPLEXITY_API_KEY environment variable for API calls.
```

### Config Section

```toml
# ~/.ash/config.toml

[skills.research]
PERPLEXITY_API_KEY = "pplx-..."  # Direct match - injected as $PERPLEXITY_API_KEY
model = "haiku"                   # Override skill's default model
enabled = true                    # Can disable without removing file

[skills.code-review]
enabled = false                   # Disabled
```

Config keys match env var names exactly (UPPER_CASE). No case conversion.

### System Prompt Listing

Skills are listed with name and description only:

```markdown
## Skills

Use the `use_skill` tool to invoke a skill with context.

- **research**: Research topics using Perplexity AI
- **code-review**: Review code for issues and improvements
```

### Tool Interface

```python
# use_skill tool
{
    "name": "use_skill",
    "input": {
        "skill": "research",
        "message": "Find the latest Python 3.13 async features",
        "context": "User is upgrading a Django app from 3.11"
    }
}

# Returns
{
    "content": "Python 3.13 introduces several async improvements...",
    "iterations": 3
}
```

### CLI Commands

```bash
# Validate skill format
ash skill validate <path>

# List skills
ash skill list
```

### Python Classes

```python
@dataclass
class SkillDefinition:
    """Skill loaded from SKILL.md files."""
    name: str
    description: str
    instructions: str

    skill_path: Path | None = None

    # Provenance
    authors: list[str] = field(default_factory=list)  # Who created/maintains this skill
    rationale: str | None = None                       # Why this skill was created

    # Subagent execution
    env: list[str] = field(default_factory=list)           # Env vars to inject
    packages: list[str] = field(default_factory=list)      # System packages (apt)
    allowed_tools: list[str] = field(default_factory=list) # Tool whitelist
    model: str | None = None                                # Model override
    max_iterations: int = 10                                # Iteration limit
```

```python
class SkillConfig(BaseModel):
    """Per-skill configuration."""
    model: str | None = None
    enabled: bool = True

    class Config:
        extra = "allow"  # Allow UPPER_CASE env var fields

    def get_env_vars(self) -> dict[str, str]:
        """Get env vars (extra fields with UPPER_CASE names)."""
        ...
```

### Registry

```python
class SkillRegistry:
    def discover(self, workspace_path: Path) -> None:
        """Load skills from workspace directory."""
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

    def validate_skill_file(self, path: Path) -> tuple[bool, str | None]:
        """Validate a skill file format without loading."""
        ...
```

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|
| `use_skill("research", ...)` | Spawns subagent, returns result | Isolated LLM loop |
| Skill with `env: [FOO]` | FOO injected from config | `[skills.x].FOO = "..."` |
| Skill with `packages: [jq]` | jq installed in sandbox | Via apt-get at build |
| Skill with `allowed_tools` | Subagent restricted to those tools | Empty = all tools |
| Skill with `model: haiku` | Uses haiku model | Config can override |
| Skill with config `enabled = false` | Filtered from prompt | Not invocable |
| `ash skill list` | Shows registered skills | |

## Errors

| Condition | Response |
|-----------|----------|
| Skill not found | `use_skill` returns error |
| Skill disabled | `use_skill` returns error |
| Missing env var in config | Skill runs without that var (warning logged) |
| Max iterations exceeded | Returns partial result with error flag |
| Tool not in allowed_tools | Subagent tool call blocked with error |

## Dependencies

Skills can declare dependencies in three ways:

### System Packages

Use the `packages:` field for system binaries (installed via apt at sandbox build):

```yaml
---
packages:
  - jq
  - ffmpeg
  - curl
---
```

### Python Dependencies (PEP 723)

For Python scripts, declare dependencies inline using PEP 723:

```python
# /// script
# dependencies = ["requests>=2.28", "pandas"]
# ///

import requests
import pandas as pd
# ...
```

Run with `uv run script.py` - dependencies are resolved automatically.

### CLI Tools (uvx)

For Python CLI tools, use `uvx` to run them without installation:

```bash
uvx ruff check .
uvx black --check file.py
```

| Need | Solution |
|------|----------|
| System binary (jq, ffmpeg) | `packages: [jq, ffmpeg]` |
| Python library to import | PEP 723 in script |
| Python CLI tool to run | `uvx toolname` |

## Verification

```bash
uv run pytest tests/test_skills.py -v
uv run pytest tests/test_skill_execution.py -v

# Manual testing
# 1. Create a skill that needs an env var
mkdir -p workspace/skills/test-api
cat > workspace/skills/test-api/SKILL.md << 'EOF'
---
description: Test API key injection
env:
  - TEST_API_KEY
allowed_tools: [bash]
---

Echo the TEST_API_KEY environment variable to verify injection.
Run: echo "Key: $TEST_API_KEY"
EOF

# 2. Configure the env var
# Add to config.toml:
# [skills.test-api]
# TEST_API_KEY = "test-secret-123"

# 3. Test invocation via chat
uv run ash chat
> use the test-api skill to check if API key is available

# Should see "Key: test-secret-123" in output
```

- Skills loaded from workspace/skills/
- Skills listed in system prompt (name + description only)
- `use_skill` tool invokes skill as subagent
- Env vars injected from config
- Tool restrictions enforced
- Model override works
- Unavailable skills filtered
- CLI commands work
