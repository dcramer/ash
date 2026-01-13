"""Write-skill subagent - creates high-quality skills.

This module handles the write-skill subagent which creates skills by:
- Researching API documentation
- Writing the SKILL.md file with proper format
- Validating the output before finishing
"""

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash.skills.base import SubagentConfig
    from ash.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

# Subagent configuration
ALLOWED_TOOLS = ["web_search", "web_fetch", "write_file", "read_file", "bash"]
MAX_ITERATIONS = 15

# Mandatory skill template - this exact format must be followed
SKILL_TEMPLATE = """
---
description: One-line description starting with a verb (under 80 chars)
config:
  - API_KEY
required_tools:
  - bash
---

Brief explanation of what this skill does.

## Implementation

```bash
curl -sfS "https://api.example.com/endpoint?key=$SKILL_API_KEY" | jq '.results'
```
""".strip()

# Validation checklist - must pass before finishing
VALIDATION_CHECKLIST = """
## Validation Checklist

Before finishing, read back the file and verify ALL of these:

- [ ] File path is `/workspace/skills/<name>/SKILL.md`
- [ ] File starts with `---` (YAML frontmatter delimiter)
- [ ] Has `description:` field (required, under 80 chars)
- [ ] API keys use `config: [API_KEY]` (not custom names like `MUNI_API_KEY`)
- [ ] Bash commands use `$SKILL_API_KEY` (not `$MUNI_API_KEY` or other custom vars)
- [ ] No emoji characters anywhere in the file
- [ ] Any scripts are in `scripts/` subdirectory, not skill root

If any check fails, fix the file before reporting success.
""".strip()

# Directory structure guide
DIRECTORY_STRUCTURE = """
## Directory Structure

```
/workspace/skills/<skill-name>/
├── SKILL.md           # Required: frontmatter + instructions
├── scripts/           # Optional: bash/python scripts
│   └── check.sh
├── references/        # Optional: docs loaded via read_file
└── assets/            # Optional: templates, data files
```

**Critical paths:**
- SKILL.md goes at: `/workspace/skills/<name>/SKILL.md`
- Scripts go at: `/workspace/skills/<name>/scripts/<script>.sh`

**Do NOT:**
- Write to `/workspace/<name>.md` (missing `skills/` directory)
- Write to `/workspace/workspace/...` (double workspace)
- Put scripts directly in skill root (use `scripts/` subdir)
""".strip()


def build_write_skill_prompt(
    goal: str,
    skill_name: str | None = None,
    existing_skill: str | None = None,
) -> str:
    """Build the system prompt for write-skill subagent.

    Structured as:
    1. Template (mandatory format)
    2. Process (research → write → validate → report)
    3. Checklist (verification before finishing)
    4. Directory structure
    5. Task (goal + existing skill if updating)

    Args:
        goal: What the skill should accomplish.
        skill_name: Optional suggested skill name.
        existing_skill: Content of existing skill to update (None for new skills).

    Returns:
        Complete system prompt.
    """
    parts = []

    # 1. Header with mandatory template
    parts.append(f"""# Skill Writer

You create SKILL.md files. Every skill **must** follow this exact format:

```markdown
{SKILL_TEMPLATE}
```

## Critical Rules

1. **YAML frontmatter is mandatory** - file MUST start with `---`
2. **Use `config: [API_KEY]`** for API keys - NOT custom names
3. **Use `$SKILL_API_KEY`** in bash - the system adds the `SKILL_` prefix
4. **No emoji** - plain text only
5. **Scripts in `scripts/`** - not in skill root directory""")

    # 2. Process
    parts.append("""## Process

1. **Research** - Find API documentation (web_search, web_fetch)
2. **Write** - Create SKILL.md with proper frontmatter at the correct path
3. **Validate** - Read the file back and run through the checklist below
4. **Report** - Tell user what was created and any setup needed (e.g., get API key)

### When to Stop

Stop and report to user if:
- API requires authentication you don't have
- No working public API exists after 2-3 search attempts
- API is down or rate-limited

Do NOT keep trying different approaches. Report what's blocking you.""")

    # 3. Validation checklist
    parts.append(VALIDATION_CHECKLIST)

    # 4. Directory structure
    parts.append(DIRECTORY_STRUCTURE)

    # 5. Task
    task_parts = ["## Your Task"]

    if existing_skill:
        task_parts.append(f"""
**Mode:** UPDATE existing skill
**Skill name:** `{skill_name}` (do not change)
**Path:** `/workspace/skills/{skill_name}/SKILL.md`
**Goal:** {goal}

### Current Skill Content (broken - needs fixing)

```markdown
{existing_skill}
```

Fix this skill to match the required format. Common issues:
- Missing YAML frontmatter (`---` delimiters)
- Custom config names instead of `API_KEY`
- Custom env vars instead of `$SKILL_API_KEY`
- Emoji characters
- Scripts in wrong location""")
    else:
        if skill_name:
            task_parts.append(f"""
**Skill name:** `{skill_name}` (use this exact name)
**Path:** `/workspace/skills/{skill_name}/SKILL.md`""")
        else:
            task_parts.append("""
**Choose a skill name** - lowercase, hyphens only (e.g., `check-muni`, `search-github`)""")

        task_parts.append(f"""
**Goal:** {goal}

Hardcode specific details (stop IDs, routes, etc.) rather than making generic parameterized skills.""")

    parts.append("\n".join(task_parts))

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
        input_data: Input containing:
            - goal: What the skill should accomplish (required)
            - name: Skill name (optional)
            - existing_skill: Content of existing skill for updates (optional)
        **kwargs: Extra context from executor (tool_definitions, workspace_path).

    Returns:
        SubagentConfig ready for execution.

    Raises:
        ValueError: If required input is missing.
    """
    from ash.skills.base import SubagentConfig

    goal = input_data.get("goal")
    if not goal:
        raise ValueError("Missing required input: goal")

    skill_name = input_data.get("name")
    existing_skill: str | None = input_data.get("existing_skill")

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
        existing_skill=existing_skill,
    )

    # Adjust initial message based on mode
    if existing_skill:
        initial_message = (
            "Update the skill to fix format issues. "
            "Read it back after writing to verify the checklist passes."
        )
    else:
        initial_message = (
            "Create the skill following the process: research, write, validate, report. "
            "Read the file back after writing to verify the checklist passes."
        )

    return SubagentConfig(
        system_prompt=system_prompt,
        allowed_tools=ALLOWED_TOOLS,
        max_iterations=MAX_ITERATIONS,
        initial_message=initial_message,
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
