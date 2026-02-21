# /audit-prompt

Audit the main agent's system prompt against best practices.

## Usage

```
/audit-prompt
```

## Process

1. Read the spec: `specs/agent-prompts.md` (System Prompt section)
2. Read the implementation: `src/ash/core/prompt.py`
3. Check each requirement from the spec
4. Report findings with specific line references

## Checklist

### Section Ordering

Verify sections appear in the correct order in `build()`:

1. Soul
2. Core Principles (critical constraints near top)
3. Tools
4. Skills
5. Agents
6. Model Aliases
7. Workspace
8. Sandbox
9. Runtime
10. Current Message (sender context)
11. Known People
12. Memory
13. Conversation Context
14. Session

### Core Principles

Check for a dedicated section with critical behavioral constraints:

- [ ] NEVER claim success without verification
- [ ] NEVER attempt task after agent fails
- [ ] ALWAYS use tools for lookups
- [ ] Report failures explicitly

### Parallel Execution Guidance

Check Tools section includes:

- [ ] Guidance on parallel tool execution
- [ ] Example of when to parallelize
- [ ] When to run sequentially (dependencies)

### Error Recovery Patterns

Check Tools section includes:

- [ ] Timeout handling guidance
- [ ] Rate limit handling
- [ ] Persistent failure escalation

### Result Visibility

Check each section that uses tools/skills/agents states:

- [ ] Results are not visible to user
- [ ] Must summarize results in response
- [ ] Don't react without showing content

### Formatting Consistency

Check across all sections:

- [ ] Major sections use `##`
- [ ] Subsections use `###`
- [ ] Rules use bullet points
- [ ] Commands use code blocks

### Context-Aware Sections

Verify conditional sections:

- [ ] Sender section only for group chats
- [ ] Scheduled task guidance varies from interactive
- [ ] Fresh vs persistent session guidance differs

## Output Format

```
## System Prompt Audit

**Result: PASS | PARTIAL | FAIL**

### Section Ordering
| Expected | Actual | Status |
|----------|--------|--------|
| ... | ... | PASS/FAIL |

### Core Principles
| Requirement | Status | Location |
|-------------|--------|----------|
| ... | PASS/MISSING | line X |

### Parallel Execution
| Requirement | Status | Location |
|-------------|--------|----------|
| ... | PASS/MISSING | line X |

### Error Recovery
| Requirement | Status | Location |
|-------------|--------|----------|
| ... | PASS/MISSING | line X |

### Result Visibility
| Section | States invisible | Status |
|---------|------------------|--------|
| Tools | Yes/No | PASS/FAIL |
| Skills | Yes/No | PASS/FAIL |
| Agents | Yes/No | PASS/FAIL |

### Formatting
| Issue | Location | Recommendation |
|-------|----------|----------------|
| ... | line X | ... |

### Recommendations
1. [High priority issues]
2. [Medium priority issues]
3. [Low priority issues]
```

## Rules

- Reference specific line numbers in findings
- Check actual generated prompt, not just code structure
- Prioritize findings by impact on agent behavior
- Include concrete recommendations for each issue
