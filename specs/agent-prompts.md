# Agent Prompt Design

> Patterns for writing effective agent and skill prompts

Files: src/ash/agents/builtin/plan.py, src/ash/agents/builtin/research.py, src/ash/agents/builtin/skill_writer.py, src/ash/tools/builtin/skills.py

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
- Include response length guidance (aim for <300 chars for simple queries)

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

## Structured Content Formatting

When prompts or tool results contain concrete sections, use HTML-like tags to delineate them. This helps the LLM parse content boundaries accurately.

### Why Tags?

Based on [Anthropic's XML tag guidance](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/use-xml-tags):

- **Clarity**: Separate different parts unambiguously
- **Accuracy**: Reduce misinterpretation (e.g., examples vs instructions)
- **Parseability**: Easy to extract sections post-processing
- **Training alignment**: Claude was trained on structured prompts

### Tag Style

Use plain tags for each section - no outer wrapper needed:

```
<instruction>
This is the result from running the "translator" skill.
The user has NOT seen this output. Interpret and include it in your response.
</instruction>
<output>
La traducción es: "Hola mundo"
</output>
```

Keep it minimal - just tag each distinct section.

### When to Use Tags

| Scenario | Tags | Example |
|----------|------|---------|
| Subagent results | `<instruction>`, `<output>` | Skill and agent tool results |
| Multi-part prompts | `<instructions>`, `<context>`, `<examples>` | Complex system prompts |
| Chain of thought | `<thinking>`, `<answer>` | Structured reasoning |
| Code with explanation | `<code>`, `<explanation>` | Teaching contexts |

### Implementation

Files: `src/ash/tools/base.py` (shared), `src/ash/tools/builtin/skills.py`, `src/ash/tools/builtin/agents.py`

```python
# Shared function in tools/base.py
def format_subagent_result(content: str, source_type: str, source_name: str) -> str:
    return f"""<instruction>
This is the result from the "{source_name}" {source_type}.
The user has NOT seen this output. Interpret and include relevant parts in your response.
</instruction>
<output>
{content}
</output>"""

# Thin wrappers in skills.py and agents.py
def format_skill_result(content: str, skill_name: str) -> str:
    return format_subagent_result(content, "skill", skill_name)
```

## Result Visibility

### MUST

- State "Results are not visible to the user" in Tools, Skills, and Agents sections
- After tool/skill/agent use, present the results directly. Don't narrate the process unless it was complex.
- Include the actual content when the user asked for content (jokes, searches, file reads)

### MUST NOT

- React to content without showing it ("that's funny!" without the joke)
- Say "done", "here it is", or "I ran the skill" without actual output
- Assume the user saw what you saw

### Silent Response Sentinel

In passive engagement (group chats where the bot wasn't directly mentioned), the agent may respond with exactly `[NO_REPLY]` to indicate it has nothing useful to add. The provider layer detects this and suppresses message delivery entirely.

- Only meaningful in passive engagement contexts
- Must be the entire response (no surrounding text)
- Provider deletes thinking message and skips response delivery
- User message is still persisted; assistant message is not

## Prompt Structure

### MUST

- Place critical rules near the top (high attention zone)
- Repeat key rules in each relevant section (Tools, Skills, Agents) - don't cross-reference
- Keep rules concise - one sentence > verbose paragraph

### SHOULD

- Explain WHY a rule exists (helps model generalize)
- Use normal language with Claude 4.x (not "CRITICAL", "MUST" everywhere)
- Match prompt style to desired output style

### MUST NOT

- Bury critical rules in nested sub-sections
- Use cross-references ("see Tools section for result handling")
- Add more words to fix a problem - simplify first

## Claude 4.x Considerations

### Behaviors to Know

- Skips verbal summaries after tool calls by default
- More responsive to instructions - dial back aggressive language
- Pays close attention to examples - ensure they match desired behavior

### Recommended Patterns

- "After completing a task that involves tool use, provide a quick summary"
- Explicit action instructions ("Change X" not "Can you suggest changes to X")
- State tool/skill results are invisible in each section that uses them

## System Prompt (Main Agent)

Files: src/ash/core/prompt.py

The main agent's system prompt is built dynamically by `SystemPromptBuilder`. This section covers requirements and patterns for the system prompt.

### Section Ordering

Sections MUST appear in this order (critical rules first, context last):

1. **Soul** - Personality and voice
2. **Core Principles** - Critical behavioral constraints (near top for attention)
3. **Available Tools** - Tool list + usage guidance
4. **Skills** - Skill list + invocation guidance
5. **Agents** - Agent list + delegation patterns
6. **Model Aliases** - Available model configurations
7. **Workspace** - Working directory info
8. **Sandbox** - Execution environment + CLI commands
9. **Runtime** - Model, timezone, current time
10. **Current Message** - Sender context (group chats only)
11. **Known People** - User's contacts
12. **Memory** - Retrieved context + memory guidance
13. **Conversation Context** - Time gap awareness
14. **Session** - History file access

### Core Principles Section

Place critical behavioral constraints immediately after the soul. These are the highest-priority rules that should never be violated.

```markdown
## Core Principles

- NEVER claim success without verification
- NEVER attempt a task yourself after an agent fails - report and ask user
- ALWAYS use tools for lookups - never assume or guess answers
- Report failures explicitly with actual error messages
```

### Parallel Execution Guidance

Add to the Tools section:

```markdown
### Parallel Execution

When multiple independent operations are needed, execute them in parallel.
For example: reading 3 files → run 3 read_file calls simultaneously.
Only run sequentially when outputs depend on previous results.
```

### Proactive Behavior Defaults

Add to the Tools or Core Principles section:

```markdown
### Behavioral Defaults

- Implement changes rather than only suggesting them
- If intent is unclear, infer the most useful action and proceed
- Use tools to discover missing details rather than asking
```

### Error Recovery Patterns

Add to the Tools section:

```markdown
### Error Recovery

- If a command times out, report it and try a simpler approach
- If you hit rate limits, wait briefly and retry once
- For persistent failures, explain what was tried and ask user
```

### Context-Aware Sections

The prompt adapts based on context:

| Context | Sections Affected | Behavior |
|---------|-------------------|----------|
| Group chat | Current Message | Shows sender, chat title, participant paths |
| DM | Current Message | Section omitted |
| Scheduled task | Sandbox | Omits reminder commands, adds execution guidance |
| Interactive | Sandbox | Full reminder/scheduling guidance |
| Fresh session | Session | Emphasizes file reading, warns about empty assumption |
| Persistent session | Session | Lighter file guidance |

### Token Budget

- `system_prompt_buffer` in `AgentConfig` reserves 8000 tokens for system prompt
- Use conditional sections to minimize prompt size
- Avoid verbose prose - prefer scannable bullets
- Test prompt size with `len(prompt) // 4` as rough token estimate

### Formatting Standards

| Element | Format |
|---------|--------|
| Major sections | `##` header |
| Subsections | `###` header |
| Rules/constraints | Bullet points |
| Commands/examples | Code blocks |
| Key terms | **Bold** |

### Verification

```bash
# Review prompt builder
cat src/ash/core/prompt.py

# Check section ordering in build()
grep "_build_.*_section" src/ash/core/prompt.py
```

- All required sections are present
- Sections appear in correct order
- Core Principles section exists and is near top
- Parallel execution guidance is in Tools section
- Context-aware sections are conditionally included
- Formatting is consistent across sections
