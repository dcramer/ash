# Skills

> Reusable instructions that the agent reads and follows

Files: src/ash/skills/base.py, src/ash/skills/registry.py, src/ash/cli/commands/skill.py

## Overview

Skills are markdown files with YAML frontmatter containing instructions the agent reads and follows. There is no execution machinery - the agent reads the SKILL.md file directly and uses its available tools to follow the instructions.

This is the "pi-mono model" of skills: knowledge packages that augment the agent's capabilities without separate execution contexts.

**Note:** For complex multi-step tasks like research, use Agents instead (see specs/agents.md).

## Requirements

### MUST

- Load workspace skills from `workspace/skills/`
- Support directory format: `skills/<name>/SKILL.md` (preferred)
- Support flat markdown: `skills/<name>.md` (convenience)
- Support pure YAML: `skills/<name>.yaml` (backward compatibility)
- Each skill defines: name, description, instructions
- Support skill requirements: bins, env, os filtering
- Filter unavailable skills from system prompt and iteration
- List skills in system prompt with file paths
- Provide CLI commands for skill management
- Default skill name to directory/filename if not specified

### SHOULD

- Support optional `required_tools` field (documentation only)
- Support optional `input_schema` field (documentation only)
- Log skill discovery with count

### MAY

- Watch workspace/skills/ for changes and hot-reload
- Track skill usage statistics

## Interface

### Directory Skill Format (Preferred)

```
workspace/skills/
  summarize/
    SKILL.md
    scripts/              # Optional: referenced scripts
      main.py
  explain/
    SKILL.md
```

```markdown
<!-- workspace/skills/summarize/SKILL.md -->
---
description: Summarize text or documents concisely
required_tools:
  - bash
requires:
  bins:
    - pandoc
  os:
    - linux
    - darwin
input_schema:
  type: object
  properties:
    content:
      type: string
      description: Text or file path to summarize
  required:
    - content
---

You are a summarization assistant. Create clear, concise summaries.

Extract key points only. Maintain factual accuracy.
```

### System Prompt Format

Skills are listed in the system prompt with their file paths:

```markdown
## Skills

Skills provide task-specific instructions.
Read a skill's file when the task matches its description.

### Available Skills

- **summarize**: Summarize text or documents concisely
  File: /workspace/skills/summarize/SKILL.md
```

The agent reads the SKILL.md file using `read_file` when a task matches the skill description.

### CLI Commands

```bash
# Scaffold new skill
ash skill init <name> [--resources scripts,references,assets]

# Validate skill format
ash skill validate <path>

# List skills (with availability status)
ash skill list [--all]

# Reload skills (after manual creation)
ash skill reload
```

### Environment Variables

Skills that need environment variables should reference vars from the central `[env]` section in `~/.ash/config.toml`:

```toml
# ~/.ash/config.toml
[env]
MUNI_API_KEY = "abc123"
BRAVE_API_KEY = "xyz789"
GITHUB_TOKEN = "$GITHUB_TOKEN"  # Reference actual env var
```

Skills reference these directly in their instructions:

```bash
# In SKILL.md
curl "https://api.example.com?key=$MUNI_API_KEY"
```

No `SKILL_*` prefix needed - variables are loaded into the session environment at startup.

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
    """Skill loaded from SKILL.md files."""
    name: str
    description: str
    instructions: str
    required_tools: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    requires: SkillRequirements = field(default_factory=SkillRequirements)
    skill_path: Path | None = None  # Path to skill directory

    def is_available(self) -> tuple[bool, str | None]:
        """Check if skill is available on current system."""
        return self.requires.check()
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

    def __iter__(self) -> Iterator[SkillDefinition]:
        """Iterate over available skills only."""
        ...

    def __len__(self) -> int: ...
```

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|
| Registry.discover() called | Workspace skills loaded | From workspace/skills/ |
| Skills discovered | Listed in system prompt with paths | Agent reads file to follow |
| Task matches skill description | Agent reads SKILL.md | Follows instructions directly |
| Empty workspace/skills/ | No skills | No error |
| Skill without `name` in frontmatter | Uses directory/filename | e.g., `summarize/SKILL.md` → `summarize` |
| Skill with `requires.bins` not in PATH | Filtered from prompt/iteration | Still registered |
| Skill with `requires.env` not set | Filtered from prompt/iteration | Still registered |
| Skill with `requires.os` not matching | Filtered from prompt/iteration | Still registered |
| `ash skill init foo` | Creates skills/foo/SKILL.md template | Scaffolding |
| `ash skill validate path` | Validates format | Returns errors if invalid |
| `ash skill list` | Shows available skills | With availability status |

## Errors

| Condition | Response |
|-----------|----------|
| Skill not found | KeyError from registry.get() |
| Missing frontmatter | Logged warning, skill skipped during discovery |
| Missing description | Logged warning, skill skipped |
| Empty instructions | Logged warning, skill skipped |
| Invalid YAML frontmatter | Logged warning, skill skipped |

## Verification

```bash
uv run pytest tests/test_skills.py -v

# Test skill creation via CLI
uv run ash skill init greeting --description "Greet the user warmly"
# Creates workspace/skills/greeting/SKILL.md

# Test skill validation
uv run ash skill validate workspace/skills/greeting/SKILL.md
# Should report valid or list errors

# Test skill list
uv run ash skill list
# Shows available skills

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
uv run ash skill list
```

- Workspace skills loaded from workspace/skills/
- Skills listed in system prompt with file paths
- Agent reads SKILL.md directly to follow instructions
- Directory format `<name>/SKILL.md` loads correctly
- Flat markdown files still supported
- YAML files still supported (backward compatibility)
- Invalid files skipped with warning
- Skills with unmet requirements filtered from prompt
- CLI commands work for init/validate/list/reload
