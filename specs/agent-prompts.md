# Agent Prompt Design

> Patterns for writing effective agent and skill prompts

Files: src/ash/agents/builtin/*.py, src/ash/tools/builtin/skills.py

## Requirements

### MUST

- Start with clear role statement ("You are a X" not "You help with X")
- Include explicit process/steps for the task
- Define what to do when things fail
- List concrete behaviors to NEVER do
- Specify output format/what to report back

### SHOULD

- Use `**IMPORTANT**` / `**CRITICAL**` for key rules
- Give concrete examples of bad behavior to avoid
- Explain WHY constraints exist when non-obvious
- Group related guidance under `##` headers

### MUST NOT

- Use passive voice for role ("You help..." → "You are...")
- Leave error handling undefined
- Assume success - always handle failure paths
- Create verbose prose - be direct and scannable

## Interface

### Prompt Structure

```
[Role Statement]
You are a [specific role]. Your job is to [primary task].

## [Main Process Section]
1. Step one
2. Step two
...

## Handling Errors
When something fails:
- [Specific guidance]
- [What NOT to do]

**NEVER do any of the following:**
- [Concrete bad behavior 1]
- [Concrete bad behavior 2]

## Output
[What to report back, format requirements]

---
[Task-specific instructions below separator, for skill agents]
```

### Skill Agent Wrapper

Skills use `SKILL_AGENT_WRAPPER` in `src/ash/tools/builtin/skills.py`:
- Prepended to all skill system prompts
- Contains role, error handling, and output guidance
- Individual skills only need task-specific instructions

### Built-in Agents

Built-in agents define their own `*_PROMPT` constants:
- `SKILL_WRITER_PROMPT` in `skill_writer.py`
- `RESEARCH_SYSTEM_PROMPT` in `research.py`

## Behaviors

| Component | Prompt Source | Error Handling |
|-----------|---------------|----------------|
| Skill agent | SKILL_AGENT_WRAPPER + SKILL.md | Stop on error, report to user |
| Skill writer | SKILL_WRITER_PROMPT | Fix and retry validation |
| Research agent | RESEARCH_SYSTEM_PROMPT | Continue with available sources |

## Examples

### Good Role Statement

```
You are a skill builder. You create SKILL.md files that define specialized behaviors.
```

### Bad Role Statement

```
You help create and update SKILL.md files for Ash skills.
```

### Good Error Handling

```
When a command fails or returns an error:
- Report the error message to the user
- STOP - do not attempt to fix, debug, or work around the problem

**NEVER do any of the following:**
- Read the script source to understand why it failed
- Copy or modify script files
- Use sed, awk, or other tools to edit files
```

### Bad Error Handling

```
(none - the prompt doesn't mention what to do on failure)
```

## Errors

| Condition | Response |
|-----------|----------|
| Prompt lacks role statement | Agent behavior is ambiguous |
| Prompt lacks error handling | Agent goes rogue trying to "help" |
| Prompt lacks NEVER rules | Agent invents undesirable behaviors |
| Prompt lacks output format | Reports are inconsistent |

## Verification

```bash
# Review each agent prompt
grep -l "PROMPT" src/ash/agents/builtin/*.py

# Check skill agent wrapper
grep -A 30 "SKILL_AGENT_WRAPPER" src/ash/tools/builtin/skills.py
```

- All agents have clear role statements
- All agents have error handling sections
- All agents have NEVER rules for common bad behaviors
- All agents specify output format
- SKILL_AGENT_WRAPPER covers execution, errors, and output
