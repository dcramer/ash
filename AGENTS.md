# Ash

Personal assistant agent with sandboxed tool execution.

## Quick Reference

- **Specs**: See `SPECS.md` and `specs/*.md` for feature specifications
- **Architecture**: See `ARCHITECTURE.md` for tech stack and roadmap
- **Config**: See `config.example.toml` for all options

## Package Manager

Use **uv**: `uv sync --all-groups`, `uv run pytest`, `uv run ruff check --fix .`

## Commands

```bash
uv run ash chat               # Interactive CLI chat
uv run ash serve              # Start server
uv run ash upgrade            # Run migrations, check sandbox
uv run ash sandbox build      # Build sandbox image
uv run ash sandbox verify     # Run security tests
```

## Conventions

- Async everywhere (`async def`, `await`)
- Type hints required
- Pydantic for validation
- ABC for interfaces in `*/base.py`
- Tests in `tests/`

## Verification

Always verify changes with appropriate methods:
1. **Unit tests**: `uv run pytest tests/ -v`
2. **CLI testing**: Test user-facing changes via `uv run ash chat "prompt"`
3. **Sandbox verification**: `uv run ash sandbox verify` for security tests

## Tools

Available tools for the agent (all execute in Docker sandbox):

| Tool | Purpose | Config Required |
|------|---------|-----------------|
| `bash` | Execute shell commands | `[sandbox]` |
| `web_search` | Search web via Brave API | `[brave_search]` + `network_mode: bridge` |

Tools are registered at startup based on configuration. See `config.example.toml`.

## Commit Attribution

AI commits MUST include:
```
Co-Authored-By: (the agent model's name and attribution byline)
```

## Specifications

Every feature MUST have a spec in `specs/<feature>.md`. See `SPECS.md` for format.

### Spec Rules
- Update spec BEFORE implementing changes
- Update spec AFTER discovering new behaviors/errors
- Keep specs concise - no prose, only testable requirements
- Specs are stateless - no tracking of implementation status

### `/write-spec <feature>`

Create or update a feature specification:
1. Read project context: `CLAUDE.md`, `ARCHITECTURE.md`, existing specs
2. Read existing spec if present: `specs/<feature>.md`
3. Read implementation files to understand current state
4. Draft spec with requirements, interface, behaviors, errors, verification
5. **Review against project goals**: Does this spec serve the project's purpose (personal assistant with memory, sandboxed tools, etc.)? Does it integrate properly with other features?
6. Revise if the spec doesn't align with project objectives
7. Follow format in `SPECS.md`
8. Update `SPECS.md` index if new spec

### `/verify-spec <feature>`

Verify implementation matches specification:
1. Read spec: `specs/<feature>.md`
2. Run verification commands from spec
3. Check each requirement (MUST/SHOULD/MAY)
4. Report: PASS (all MUST + SHOULD), PARTIAL (all MUST), FAIL (missing MUST)

## Skills

Use `/commit` for commits, `/create-pr` for PRs, `/find-bugs` before merging.
