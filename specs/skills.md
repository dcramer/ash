# Skills

> Workspace-defined behaviors that orchestrate tools with model preferences

Files: src/ash/skills/base.py, src/ash/skills/registry.py, src/ash/skills/executor.py, src/ash/tools/builtin/skills.py

## Requirements

### MUST

- Load skills from YAML files in `workspace/skills/` directory
- Each skill defines: name, description, instructions, preferred_model, required_tools
- SkillRegistry discovers and loads skills from workspace
- SkillExecutor creates sub-agent loop with skill instructions as system prompt
- Expose skills to LLM via `list_skills` and `use_skill` tools
- Skills can reference model aliases (e.g., "fast", "default")
- Validate required_tools exist before skill execution
- Pass skill results back to parent agent

### SHOULD

- Support skill parameters via input_schema (JSON Schema)
- Allow skills to specify max_iterations independently
- Log skill execution with duration and iteration count
- Cache loaded YAML skills for performance
- Provide clear error when referenced model alias not found

### MAY

- Support skill chaining (one skill invoking another via use_skill)
- Watch workspace/skills/ for changes and reload
- Track skill usage statistics

## Interface

### YAML Skill Format

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
      description: Text or file path to summarize
    format:
      type: string
      enum: [bullets, paragraph, tldr]
      default: bullets
  required:
    - content
instructions: |
  You are a summarization assistant. Create clear, concise summaries.
  Extract key points only. Maintain factual accuracy.
  Use the requested format for output.
```

### Python Classes

```python
@dataclass
class SkillDefinition:
    """Skill loaded from YAML."""
    name: str
    description: str
    instructions: str
    preferred_model: str | None = None
    required_tools: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 5

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
    def discover(self, workspace_path: Path) -> None:
        """Load all YAML skills from workspace/skills/."""
        ...

    def get(self, name: str) -> SkillDefinition:
        """Get skill by name. Raises KeyError if not found."""
        ...

    def has(self, name: str) -> bool: ...

    def list(self) -> list[str]:
        """List available skill names."""
        ...

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get skill definitions for LLM."""
        ...
```

### Executor

```python
class SkillExecutor:
    def __init__(
        self,
        registry: SkillRegistry,
        tool_executor: ToolExecutor,
        model_registry: ModelRegistry,
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

### LLM Tools

```python
class ListSkillsTool(Tool):
    """List available skills from workspace."""
    name = "list_skills"
    input_schema = {"type": "object", "properties": {}}

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

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|
| `list_skills` tool call | JSON list of skill names and descriptions | |
| `use_skill(summarize, {content: "..."})` | SkillResult with summary | Sub-agent executes |
| Skill with `preferred_model: fast` | Uses `models.fast` config | Model alias resolved |
| Skill with unknown model alias | Falls back to default model | Warning logged |
| Skill requires unavailable tool | Error before execution | Validation fails |
| Skill exceeds max_iterations | Returns partial result | With limit message |
| Empty workspace/skills/ | list_skills returns empty | No error |

## Errors

| Condition | Response |
|-----------|----------|
| Skill not found | SkillResult.error("Skill 'name' not found") |
| Required tool unavailable | SkillResult.error("Skill requires tool 'bash' which is not available") |
| Invalid input schema | SkillResult.error("Invalid input: <validation error>") |
| YAML parse error | Logged warning, skill skipped during discovery |
| Model alias not found | Uses default model, logs warning |

## Verification

```bash
uv run pytest tests/test_skills.py -v
mkdir -p workspace/skills
cat > workspace/skills/test.yaml << 'EOF'
name: test
description: Test skill
instructions: Say hello
EOF
uv run ash chat "List available skills"
uv run ash chat "Use the test skill"
```

- Skills discovered from workspace/skills/
- list_skills returns available skills
- use_skill executes skill with sub-agent
- Model alias resolution works
- Missing tools detected before execution
- Invalid YAML files skipped with warning
