# Skills Specification (Condensed)

Skills are markdown files that define specialized subagents invoked via `use_skill`.

## Directory Format

```
workspace/skills/
  my-skill/
    SKILL.md          # Required - frontmatter + instructions
    references/       # Optional - loaded docs, schemas, examples
    scripts/          # Optional - Python/bash scripts for execution
    assets/           # Optional - templates, sample files
```

## SKILL.md Frontmatter

```yaml
---
description: string       # Required. One line, starts with verb, no trailing period
authors:                  # Required. Who created/maintains this skill
  - username
rationale: string         # Required. Why this skill was created (user's intent)
allowed_tools:            # Optional. Tool whitelist (empty = all tools)
  - bash
  - web_search
triggers:                 # Optional. Trigger hints/phrases (metadata only)
  - /research
env:                      # Optional. Env vars injected from [skills.<name>] config
  - API_KEY
packages:                 # Optional. System packages (apt) for sandbox
  - jq
  - curl
model: string             # Optional. Model alias override (e.g., "haiku")
max_iterations: int       # Optional. Iteration limit (default: 10)
opt_in: bool              # Optional. If true, requires explicit enablement in config
---
```

## Instructions (Markdown Body)

The markdown body after frontmatter contains instructions for the agent.

**Rules:**
- Write imperative commands, not documentation
- Reference user input with `<user_message>` or "the user's input"
- Specify expected output format
- Use full paths: `/workspace/skills/<name>/scripts/helper.py`

## Config Integration

Users configure skills in `~/.ash/config.toml`:

```toml
[skills.my-skill]
API_KEY = "secret-123"    # Injected as $API_KEY env var
model = "haiku"           # Override model
enabled = true            # Enable/disable
```

Config key names (UPPER_CASE) match `env:` field entries exactly.

## Dependencies

| Need | Solution |
|------|----------|
| System binary (jq, ffmpeg) | `packages: [jq, ffmpeg]` |
| Python library to import | PEP 723 inline metadata in script |
| Python CLI tool to run | `uvx toolname` |

### PEP 723 Example

```python
# /// script
# dependencies = ["httpx", "beautifulsoup4"]
# ///
import httpx
from bs4 import BeautifulSoup
```

Run with `uv run script.py` - dependencies resolve automatically.

## Validation

```bash
ash-sb skill validate /workspace/skills/<name>/SKILL.md
```

Checks: frontmatter exists, description present, instructions present, YAML valid.
