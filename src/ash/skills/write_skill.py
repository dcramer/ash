"""Write-skill subagent - creates high-quality skills.

This module handles the write-skill subagent which creates skills by:
- Searching for API documentation
- Reading the docs to understand endpoints
- Writing the SKILL.md file
- Testing the implementation
"""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash.skills.base import SubagentConfig
    from ash.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

# Subagent configuration
ALLOWED_TOOLS = ["web_search", "web_fetch", "write_file", "read_file", "bash"]
MAX_ITERATIONS = 15

# Path to bundled skills
BUNDLED_SKILLS_DIR = Path(__file__).parent / "bundled"

# Example skills to include (in priority order)
EXAMPLE_SKILL_NAMES = ["research", "code-review", "debug"]

# Skill schema documentation
SKILL_SCHEMA_DOCS = """
## Skill Schema

Skills are defined in `SKILL.md` files with YAML frontmatter:

```yaml
---
# === REQUIRED ===
description: string  # One-line description shown in skill list

# === EXECUTION ===
subagent: bool  # default: false
  # false: Returns instructions for main agent to follow (inline)
  # true: Runs isolated sub-agent loop with own context
model: string  # Model alias (default, fast, etc.) - default: "default"
max_iterations: int  # Max iterations for subagent mode - default: 5

# === TOOLS ===
required_tools: list[string]  # Tools the skill needs access to
  # Examples: bash, web_search, remember, recall

# === INPUT ===
input_schema:  # JSON Schema for skill inputs
  type: object
  properties:
    param_name:
      type: string
      description: What this parameter is for
  required: [param_name]

# === REQUIREMENTS ===
requires:
  bins: list[string]   # Required binaries in PATH
  env: list[string]    # Required environment variables
  os: list[string]     # Supported OS: darwin, linux, windows

# === CONFIGURATION ===
config: list[string]
  # Config values: "NAME" (required) or "NAME=default" (with default)
  # Passed to tools as SKILL_NAME environment variables
---

# Instructions (markdown body)

These become the system prompt (subagent) or returned instructions (inline).
```
""".strip()

# Validation rules
VALIDATION_RULES = """
## Validation Rules

### Name Format
- Lowercase letters, numbers, and hyphens only
- Must start with a letter
- Examples: `check-weather`, `muni-arrivals`, `code-review`

### Description
- One line, under 80 characters
- No trailing period
- Starts with a verb (Check, Search, Generate, etc.)
- Examples:
  - Good: "Check SF Muni arrival times"
  - Bad: "This skill checks Muni arrivals."

### Instructions
- Clear process with numbered steps
- Specific about what tools to use and how
- Include example commands where relevant
- Structure with markdown headers

### Execution Mode
- Use `inline` (default) for:
  - Simple documentation-style skills
  - Skills where main agent should see full context
  - Quick lookup or formatting tasks
- Use `subagent` for:
  - Multi-step tool orchestration
  - Skills needing isolated context
  - Complex iterative workflows

### Config for Secrets
- Always use `config` for API keys, tokens, and credentials
- Never hardcode secrets in instructions or scripts
- Config values become `$SKILL_<NAME>` environment variables
- **Always use `API_KEY`** as the config name for API keys
- Example: `config: [API_KEY]` -> accessible as `$SKILL_API_KEY` in bash
""".strip()

# Anti-patterns to avoid
ANTI_PATTERNS = """
## Anti-patterns to Avoid

### Vague Instructions
- Bad: "Help the user with their task"
- Good: "1. Parse the input query\\n2. Search using web_search tool\\n3. Summarize findings"

### Missing Process Structure
- Bad: "Do code review"
- Good: "## Process\\n### 1. Read the code\\n### 2. Check for bugs\\n### 3. Report findings"

### Missing Implementation Details
- Bad: "Query the API to get data" (vague description)
- Bad: Describing what to do without showing how
- Good: Include actual executable commands:
  ```bash
  curl -s "https://api.example.com/data?key=$SKILL_API_KEY" | jq '.results'
  ```
- Good: If bash is needed, set `required_tools: [bash]`

### Hardcoding Secrets or Custom Config Names
- Bad: API keys in instructions or scripts: `api_key = "abc123"`
- Bad: Custom config names: `config: [SFMTA_API_KEY]`, `config: [MY_TOKEN]`
- Good: Use standard name: `config: [API_KEY]` and reference as `$SKILL_API_KEY`

### Using Emoji
- Bad: Any emoji anywhere: "🚌", "👋", "✓"
- Good: Plain text only, no emoji characters anywhere in the skill

### Overusing Subagent Mode
- Bad: Using subagent for a simple greeting skill
- Good: Use inline for simple skills, subagent only when needed

### Generic Descriptions
- Bad: "A useful skill"
- Good: "Search git history for commits matching a pattern"

### ALL CAPS Emphasis
- Bad: "ALWAYS do X, NEVER do Y"
- Good: Use **bold** for emphasis instead

### Overly Complex Input Schema
- Bad: Deep nested objects for simple skills
- Good: Flat properties with clear descriptions
""".strip()

# Execution mode guidance
EXECUTION_MODE_GUIDANCE = """
## Choosing Execution Mode

### Use `inline` (default) when:
- The skill is primarily documentation/instructions
- The main agent should follow the steps directly
- You want the agent to have full conversation context
- The task is simple (greeting, formatting, explanations)
- Speed is important (no sub-agent overhead)

### Use `subagent` when:
- Multiple tool calls in a coordinated sequence
- The skill needs isolated context from parent conversation
- Complex multi-step workflows (research, debugging, code review)
- You want model/iteration control per-skill
- The skill should run autonomously

### Examples

Inline skills:
- `greet`: Just return a greeting message
- `explain`: Return explanation for main agent to deliver
- `summarize`: Return summary instructions

Subagent skills:
- `research`: Multiple web searches, synthesize results
- `code-review`: Read files, analyze, produce report
- `debug`: Systematic investigation with multiple tools
""".strip()


def load_example_skill(skill_name: str) -> str | None:
    """Load a bundled skill's content as an example.

    Args:
        skill_name: Name of the bundled skill.

    Returns:
        The skill's SKILL.md content, or None if not found.
    """
    skill_path = BUNDLED_SKILLS_DIR / skill_name / "SKILL.md"
    if not skill_path.exists():
        return None
    return skill_path.read_text()


def format_tool_list(tool_definitions: list[dict[str, Any]]) -> str:
    """Format available tools for inclusion in prompt.

    Args:
        tool_definitions: List of tool definition dicts with name and description.

    Returns:
        Formatted markdown list of tools.
    """
    lines = []
    for tool_def in tool_definitions:
        name = tool_def["name"]
        desc = tool_def.get("description", "")
        # Truncate long descriptions
        if len(desc) > 100:
            desc = desc[:97] + "..."
        lines.append(f"- **{name}**: {desc}")
    return "\n".join(lines)


def build_write_skill_prompt(
    goal: str,
    skill_name: str | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
    workspace_path: Path | None = None,
) -> str:
    """Build the dynamic system prompt for the write-skill subagent.

    This prompt includes:
    - Available tools from the registry
    - Skill schema documentation
    - Example skills from bundled
    - Validation rules and anti-patterns
    - The user's goal

    Args:
        goal: What the skill should accomplish.
        skill_name: Optional suggested skill name.
        tool_definitions: List of available tool definitions.
        workspace_path: Path to workspace for skill creation.

    Returns:
        Complete system prompt for the write-skill subagent.
    """
    parts = []

    # Header
    parts.append("""# Skill Writer

Create SKILL.md files (markdown with YAML frontmatter).

## Process

1. Search for API documentation
2. Read the docs to get exact endpoints and parameters
3. Write the SKILL.md
4. Test with bash to verify it works

**Never guess at APIs** - research first, then write. If you can't find docs, tell the user.

## Rules

- Output is a SKILL.md file, not Python/shell scripts
- Use `curl -sfS --compressed` for API calls
- Use `config: [API_KEY]` for secrets (accessed as `$SKILL_API_KEY`)
- No emoji
- Test before finishing""")

    # Tools that skills can reference (for the skill author to know what's available)
    if tool_definitions:
        tools_formatted = format_tool_list(tool_definitions)
        parts.append(f"""
## Available Tools

Skills can use these tools in their instructions:

{tools_formatted}""")

    # Schema documentation
    parts.append(SKILL_SCHEMA_DOCS)

    # Example skills
    examples_loaded = []
    for name in EXAMPLE_SKILL_NAMES:
        content = load_example_skill(name)
        if content:
            examples_loaded.append((name, content))

    if examples_loaded:
        parts.append(
            "\n## Example Skills\n\nStudy these examples of well-structured skills:"
        )
        for name, content in examples_loaded[:2]:  # Limit to 2 to save context
            # Truncate if too long
            if len(content) > 2000:
                content = content[:2000] + "\n... (truncated)"
            parts.append(f"\n### {name}\n\n```markdown\n{content}\n```")

    # Validation rules
    parts.append(VALIDATION_RULES)

    # Anti-patterns
    parts.append(ANTI_PATTERNS)

    # Execution mode guidance
    parts.append(EXECUTION_MODE_GUIDANCE)

    # Workspace info - always use /workspace (sandbox mount point)
    # The host workspace path is mounted at /workspace inside the sandbox
    parts.append("""
## Workspace

Skills directory: `/workspace/skills/`

Create skills in: `/workspace/skills/<skill-name>/SKILL.md`""")

    # The task
    task_parts = ["\n## Your Task"]
    if skill_name:
        task_parts.append(f"\n**Skill name:** `{skill_name}`")
        task_parts.append(f"\n**Path:** `/workspace/skills/{skill_name}/SKILL.md`")
    task_parts.append(f"\n**Goal:** {goal}")
    task_parts.append("""

Hardcode specific details (stop IDs, routes, etc.) rather than making generic parameterized skills.""")

    parts.append("".join(task_parts))

    return "\n\n---\n\n".join(parts)


# Input schema for the write-skill skill
WRITE_SKILL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {
            "type": "string",
            "description": "What the skill should accomplish",
        },
        "name": {
            "type": "string",
            "description": "Suggested skill name (optional, will be inferred if not provided)",
        },
    },
    "required": ["goal"],
}


def build_subagent_config(
    input_data: dict[str, Any],
    **kwargs: Any,
) -> "SubagentConfig":
    """Build SubagentConfig for write-skill execution.

    Args:
        input_data: Input containing 'goal' and optional 'name'.
        **kwargs: Extra context from executor:
            - tool_definitions: All available tool definitions (for showing in prompt).
            - workspace_path: Workspace path for skill creation.

    Returns:
        SubagentConfig ready for execution.

    Raises:
        ValueError: If required input is missing.
    """
    from ash.skills.base import SubagentConfig

    # Extract context from kwargs
    tool_definitions: list[dict[str, Any]] = kwargs.get("tool_definitions", [])
    workspace_path: Path | None = kwargs.get("workspace_path")

    goal = input_data.get("goal")
    if not goal:
        raise ValueError("Missing required input: goal")

    skill_name = input_data.get("name")

    # Try to extract skill name from goal if not provided
    if not skill_name and goal:
        match = re.search(r"(?:called|named)\s+['\"]([a-z0-9-]+)['\"]", goal, re.I)
        if match:
            skill_name = match.group(1).lower()
            logger.info(f"Extracted skill name '{skill_name}' from goal")

    # Build system prompt
    system_prompt = build_write_skill_prompt(
        goal=goal,
        skill_name=skill_name,
        tool_definitions=tool_definitions,
        workspace_path=workspace_path,
    )

    return SubagentConfig(
        system_prompt=system_prompt,
        allowed_tools=ALLOWED_TOOLS,
        max_iterations=MAX_ITERATIONS,
        initial_message="Create the skill according to the instructions provided.",
    )


def register(registry: "SkillRegistry") -> None:
    """Register the write-skill with the registry.

    Args:
        registry: Skill registry to register with.
    """
    registry.register_dynamic(
        name="write-skill",
        description="Create high-quality SKILL.md files",
        build_config=build_subagent_config,
        required_tools=ALLOWED_TOOLS,
        input_schema=WRITE_SKILL_INPUT_SCHEMA,
    )
