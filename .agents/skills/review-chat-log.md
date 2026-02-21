# /review-chat-log

Review a chat log or conversation transcript to identify undesirable AI assistant behaviors.

## Usage

```
/review-chat-log <path-to-log>
```

Examples:
- `/review-chat-log ~/.claude/projects/foo/abc123.jsonl`
- `/review-chat-log /tmp/conversation.json`
- `/review-chat-log ./chat-export.txt`

## Supported Formats

1. **Claude Code transcripts** (`.jsonl`) - Lines with `type`, `message`, `tool_use`, `tool_result`
2. **JSON conversations** (`.json`) - Array of messages with `role` and `content`
3. **Plain text logs** - Human/Assistant alternating format
4. **Markdown exports** - Headers denoting speakers

## Process

1. **Detect format** from file extension and content structure
2. **Parse into normalized message list** with role, content, and any tool calls
3. **Analyze for anti-patterns** (see Evaluation Criteria)
4. **Generate structured findings report**

## Evaluation Criteria

Assign severity: **CRITICAL** | **HIGH** | **MEDIUM** | **LOW**

### 1. Task Completion
- Abandoning tasks without explanation
- Claiming completion without verification
- Partial implementations presented as complete

### 2. Reasoning Quality
- Jumping to conclusions without analysis
- Ignoring contradictory evidence
- Circular reasoning or tautologies

### 3. Communication
- Over-verbose responses to simple questions
- Under-explaining complex decisions
- Repeating information unnecessarily
- Not answering the actual question asked

### 4. Instruction Following
- Ignoring explicit user requests
- Adding unrequested features or changes
- Misinterpreting clear instructions
- Scope creep beyond the request

### 5. Error Recovery
- Repeating failed approaches
- Not acknowledging mistakes
- Blaming external factors inappropriately

### 6. Context Handling
- Forgetting earlier conversation context
- Contradicting previous statements
- Asking questions already answered
- Losing track of the goal

### 7. Honesty & Accuracy
- Hallucinating facts or capabilities
- Overconfident claims without evidence
- Not admitting uncertainty when appropriate

### 8. User Experience
- Excessive caveats and disclaimers
- Unnecessary apologies
- Condescending explanations
- Not matching user's technical level

## Output Format

```
## Chat Log Review: <filename>

**Log Info:**
- Format: <detected format>
- Messages: <count> user, <count> assistant
- Tool calls: <count if applicable>

**Overall Assessment:** PASS | CONCERNS | FAIL

### Findings

#### [SEVERITY] Category: Brief Title
**Location:** Message N or timestamp
**Description:** What happened
**Evidence:** Relevant quote
**Impact:** Why this matters
**Recommendation:** How to improve

### Summary
- Critical: N | High: N | Medium: N | Low: N

### Patterns Observed
- Pattern descriptions
```

## Rules

- Focus on actionable findings, not style nitpicks
- Quote specific evidence from the log
- Consider context - some behaviors may be appropriate for the situation
- Do not expose sensitive data found in logs (API keys, passwords, PII)
- Always provide overall assessment at top for quick scanning
- Compare assistant behavior against the user's actual needs, not hypothetical ideals
