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
| [cli](specs/cli.md) | Typer-based command-line interface with consistent help behavior |
| [subsystems](specs/subsystems.md) | Modular components with clear boundaries and interfaces |
| [agent](specs/agent.md) | Agent orchestrator with agentic loop |
| [agent-prompts](specs/agent-prompts.md) | Patterns for writing effective agent and skill prompts |
| [config](specs/config.md) | Configuration loading and validation |
| [conversation-context](specs/conversation-context.md) | Smart conversation context with reply chains and gap signals |
| [llm](specs/llm.md) | LLM provider abstraction |
| [logging](specs/logging.md) | Centralized logging configuration with consistent formatting |
| [memory](specs/memory.md) | Long-term knowledge persistence across conversations |
| [models](specs/models.md) | Named model configurations with aliases |
| [sandbox](specs/sandbox.md) | Docker sandbox for command execution |
| [sentry](specs/sentry.md) | Optional error tracking and observability |
| [server](specs/server.md) | FastAPI server and webhooks |
| [skills](specs/skills.md) | Workspace-defined behaviors with model preferences |
| [telegram](specs/telegram.md) | Telegram bot integration |
| [web_fetch](specs/web_fetch.md) | Fetch and extract content from URLs |
| [web_search](specs/web_search.md) | Web search via Brave API with caching and retry |
| [research](specs/research.md) | Built-in agent for web research with synthesis |
| [rpc](specs/rpc.md) | Unix domain socket RPC for sandbox-to-host communication |
| [sessions](specs/sessions.md) | JSONL-based session persistence for conversation history |
| [workspace](specs/workspace.md) | Agent personality via SOUL.md with inheritance |
