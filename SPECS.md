# Specification System

Specs define feature requirements for implementation and verification.

## Format

Each spec is a markdown file in `specs/` with this structure:

```markdown
# Feature Name

> One-line purpose statement

Files: path/to/file.py, path/to/other.py

## Requirements

### MUST
- Requirement with testable criteria
- Another requirement

### SHOULD
- Nice-to-have with testable criteria

### MAY
- Optional behavior

## Interface

```python
# Function signatures, CLI commands, or API endpoints
def function(param: Type) -> ReturnType: ...
```

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|
| valid input | expected output | |
| edge case | expected handling | |

## Errors

| Condition | Response |
|-----------|----------|
| invalid input | Error message or behavior |

## Verification

```bash
# Commands to verify implementation
command_to_test_feature
```

- Verification check 1
- Verification check 2
```

## Rules

### MUST Include
- **Testable requirements** - Every line verifiable by running code or commands
- **Interface definition** - Exact signatures, commands, or endpoints
- **Error conditions** - What fails and how
- **Verification commands** - Specific tests to run

### MUST NOT Include
- Design rationale or "why" explanations
- Implementation suggestions or hints
- Historical context or changelog
- Future roadmap items
- Verbose prose or examples
- State tracking (specs are stateless)

### Maintenance

Specs MUST be updated when:
- Requirements change
- Interface changes
- New error conditions discovered
- Verification tests change

## Skills

- `/write-spec <feature>` - Create or update a spec
- `/verify-spec <feature>` - Run verification checks against implementation

## Index

| Spec | Description |
|------|-------------|
| [agent](specs/agent.md) | Agent orchestrator with agentic loop |
| [config](specs/config.md) | Configuration loading and validation |
| [llm](specs/llm.md) | LLM provider abstraction |
| [memory](specs/memory.md) | Persistent memory with context retrieval |
| [models](specs/models.md) | Named model configurations with aliases |
| [sandbox](specs/sandbox.md) | Docker sandbox for command execution |
| [server](specs/server.md) | FastAPI server and webhooks |
| [skills](specs/skills.md) | Workspace-defined behaviors with model preferences |
| [telegram](specs/telegram.md) | Telegram bot integration |
| [web_search](specs/web_search.md) | Web search via Brave API in sandbox |
