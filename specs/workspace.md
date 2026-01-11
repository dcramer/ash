# Workspace

> Agent personality and identity configuration via SOUL.md

Files: src/ash/config/workspace.py

## Requirements

### MUST

- Load SOUL.md from workspace directory
- Support YAML frontmatter in SOUL.md
- Support personality inheritance via `extends` frontmatter
- Provide built-in "ash" personality (Ash Ketchum inspired)
- Use default personality when no SOUL.md exists
- Parse frontmatter with yaml.safe_load
- Append custom content after base personality when extending

### SHOULD

- Log warning for unknown personality in `extends`
- List available personalities in warning message
- Handle missing frontmatter gracefully (use content as-is)
- Support custom files via load_custom_file method

### MAY

- Support additional built-in personalities
- Allow personalities from external sources
- Cache parsed workspace for performance

## Interface

### SOUL.md Format

```markdown
---
extends: ash
---

# Custom Additions

Additional personality customizations appended to base.
```

### Frontmatter Fields

| Field | Type | Description |
|-------|------|-------------|
| extends | string | Name of built-in personality to inherit |

### Built-in Personalities

| Name | Description |
|------|-------------|
| ash | Ash Ketchum inspired - enthusiastic, determined, action-oriented |

### Python Classes

```python
@dataclass
class SoulConfig:
    """Configuration parsed from SOUL.md frontmatter."""
    extends: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class Workspace:
    """Loaded workspace configuration."""
    path: Path
    soul: str = ""
    soul_config: SoulConfig = field(default_factory=SoulConfig)
    custom_files: dict[str, str] = field(default_factory=dict)

class WorkspaceLoader:
    SOUL_FILENAME = "SOUL.md"

    def __init__(self, workspace_path: Path) -> None: ...

    def load(self) -> Workspace:
        """Load workspace from directory."""
        ...

    def load_custom_file(self, filename: str, workspace: Workspace) -> str | None:
        """Load additional file from workspace."""
        ...

    def ensure_workspace(self) -> None:
        """Create workspace with default SOUL.md."""
        ...
```

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|
| SOUL.md with `extends: ash` | Base personality + custom content | Inheritance |
| SOUL.md without frontmatter | Content used as-is | No inheritance |
| No SOUL.md | Default ash personality | Fallback |
| Unknown `extends` value | Custom content only | Warning logged |
| Empty SOUL.md body with `extends` | Base personality only | Valid |

### Default SOUL.md Template

```markdown
---
extends: ash
---

# Customizations

Add your personality customizations here. They will be appended
to the base Ash personality.
```

### Ash Personality (Built-in)

```markdown
# Ash

You are Ash, a personal assistant inspired by Ash Ketchum from Pokemon.

## Personality

- Enthusiastic and determined - you never give up on helping
- Friendly and encouraging - you believe in the user's potential
- Action-oriented - you prefer doing over just talking
- Loyal and supportive - you're always on the user's side
- Curious and eager to learn - you love discovering new things

## Communication Style

- Energetic and positive tone
- Use encouraging phrases like "Let's do this!" or "We've got this!"
- Be direct and action-focused
- Ask clarifying questions when the path forward isn't clear
- Celebrate successes, no matter how small

## Catchphrases (use sparingly)

- "I choose you!" (when selecting a tool or approach)
- "Gotta catch 'em all!" (when gathering information)
- "Time to battle!" (when tackling a challenge)

## Principles

- Never give up - there's always a way
- Trust your instincts but verify with data
- Learn from every experience, success or failure
- Teamwork makes the dream work
- Respect boundaries and privacy
```

## Errors

| Condition | Response |
|-----------|----------|
| Workspace directory not found | FileNotFoundError |
| Invalid YAML frontmatter | Warning logged, content used without frontmatter |
| Unknown `extends` personality | Warning with available options, custom content only |

## Verification

```bash
uv run pytest tests/test_agent.py::TestWorkspace -v

# Test inheritance
mkdir -p workspace
cat > workspace/SOUL.md << 'EOF'
---
extends: ash
---

# Extra Rules

Always end responses with a Pokemon pun.
EOF
uv run ash chat "Hello"

# Test no inheritance
cat > workspace/SOUL.md << 'EOF'
# Custom Bot

You are a serious business assistant.
EOF
uv run ash chat "Hello"
```

- SOUL.md loaded from workspace
- Frontmatter parsed correctly
- Personality inheritance works
- Custom content appended after base
- Default personality used when no SOUL.md
- Unknown extends logs warning
