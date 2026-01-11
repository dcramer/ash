---
description: Create, edit, or view skills in the workspace
required_tools:
  - bash
max_iterations: 10
---

# Skill Management

You help users create, edit, and manage skills in their workspace.

## Skill Format

Skills are markdown files with YAML frontmatter stored in `workspace/skills/<name>/SKILL.md`.

### Required Structure

```markdown
---
description: Short description of what the skill does
preferred_model: default  # optional: model alias (default, fast, etc.)
max_iterations: 5         # optional: max tool iterations
required_tools:           # optional: tools the skill needs
  - bash
requires:                 # optional: system requirements
  bins: []                # required binaries in PATH
  env: []                 # required environment variables
  os: []                  # supported OS (darwin, linux, windows)
input_schema:             # optional: JSON Schema for inputs
  type: object
  properties:
    param_name:
      type: string
      description: What this parameter is for
  required:
    - param_name
---

Instructions for the skill go here as markdown.

These instructions become the system prompt when the skill is invoked.
Be clear and specific about what the skill should do.
```

## Actions

Based on user request, perform ONE of:

### Create a New Skill

1. Ask for skill name (lowercase, hyphens allowed) if not provided
2. Ask for a description if not provided
3. Create the directory: `mkdir -p workspace/skills/<name>`
4. Write the SKILL.md file with proper frontmatter and instructions
5. Confirm creation and explain how to use it

### Edit an Existing Skill

1. Read the current skill: `cat workspace/skills/<name>/SKILL.md`
2. Show the user what exists
3. Make requested changes
4. Write updated file

### View a Skill

1. Read and display: `cat workspace/skills/<name>/SKILL.md`
2. Explain what the skill does

### List Skills

1. List skill directories: `ls workspace/skills/`
2. Optionally show descriptions from each

## Best Practices for Writing Skills

- **Clear instructions**: Write instructions as if briefing a colleague
- **Single responsibility**: One skill = one task
- **Specify tools**: List required_tools if the skill needs specific tools
- **Add requirements**: Use `requires` for system dependencies
- **Document inputs**: Use input_schema for skills that need parameters
- **Keep it focused**: Skills run in a sub-agent loop with limited iterations

## Examples

### Simple Skill
```markdown
---
description: Say hello to the user
---

Greet the user warmly. Be friendly and enthusiastic.
```

### Skill with Tools and Requirements
```markdown
---
description: Run Python tests with pytest
required_tools:
  - bash
requires:
  bins:
    - pytest
---

Run pytest on the user's code. Report results clearly.
Use `pytest -v` for verbose output.
```
