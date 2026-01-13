# Ash

Personal assistant agent with sandboxed tool execution.

## Project Principles

### Simplicity Wins

**Removing code is always a win.** Code that has become redundant, adds unnecessary complexity, or duplicates functionality elsewhere should be removed. Every line of code is a liability - it needs to be maintained, tested, and understood. When we find dead code, unused abstractions, or over-engineered solutions, we delete them without hesitation.

### Shell and Filesystem First

**The shell is the agent's natural interface.** When designing tools and agentic behaviors, prefer shell commands and filesystem operations over custom implementations. The Unix toolchain - `grep`, `find`, `sed`, `awk`, `curl`, `jq` - represents decades of refinement. Wrap it, don't replace it.

**Design principles for toolchain:**
- **Prefer shell execution** over custom tool implementations. If `grep` can do it, use `grep`.
- **Read/write files** rather than maintain in-memory state. Files are inspectable, debuggable, and survive restarts.
- **Use standard formats** - JSONL, JSON, Markdown, TOML. Tools that work with text work with these.
- **Sandbox execution** provides security without reinventing command execution.

**When to use what:**
- **Shell tools:** File operations, text processing, system commands, git, package managers
- **Plain files:** Session transcripts (JSONL), configuration (TOML), identity (Markdown), skills, events
- **SQLite:** Only for data needing vector search, complex queries, or transactions (memories, embeddings)
- **Custom tools:** Only when shell commands genuinely can't do the job (LLM calls, structured API interactions)

The filesystem is shared state that both agent and human can inspect. `cat`, `tail -f`, `grep` become debugging tools. No special clients needed.

### Explicit Over Implicit

**No magic.** Dependencies are explicit. Configuration is explicit. When something fails, the error should make the cause obvious. Avoid clever abstractions that hide what's actually happening.

### Async All The Way

Everything is async. No blocking calls. No sync wrappers around async code. This isn't just a convention - it's how the system stays responsive during LLM calls, tool execution, and I/O.

### Types Are Documentation

Type hints aren't optional. They document intent, catch bugs before runtime, and enable tooling. Pydantic for validation at boundaries. Abstract base classes define interfaces clearly.

### Specs Before Code

New features get a spec first. The spec lives in `specs/`, documents the design decisions, and serves as the verification checklist. Update the spec before implementing, not after.

### Test What Matters

Tests exist to catch regressions and document behavior, not to hit coverage metrics. Focus on testing the contracts between components, edge cases that are easy to break, and integration points.

**What to test:**
- Core business logic (supersession, scoping, conflict detection)
- Error handling and graceful degradation
- Input validation at API boundaries
- Edge cases that are easy to break
- Integration between components

**What NOT to test:**
- Trivial CRUD operations (if SQLAlchemy breaks, we have bigger problems)
- Mock verification (testing that mocks return what you configured proves nothing)
- Dataclass constructors (Python works)
- Private methods (couples tests to implementation)
- Language features (list operations, dict access)

**Signs of bad tests:**
- Test name describes implementation, not behavior
- Test only verifies mock was called with expected args
- Test duplicates another test through a different interface
- Test would pass even if the feature was broken

## Package Manager

Use **uv**: `uv sync --all-groups`

## Quality

| Tool | Command | Purpose |
|------|---------|---------|
| ruff | `uv run ruff check --fix .` | Lint and auto-fix |
| ruff | `uv run ruff format .` | Format code |
| ty | `uv run ty check` | Type check |
| pytest | `uv run pytest` | Run tests |
| pre-commit | `pre-commit run --all-files` | Run all hooks |

## Commands

| Command | Purpose |
|---------|---------|
| `uv run ash chat` | Interactive CLI chat |
| `uv run ash serve` | Start server |
| `uv run ash upgrade` | Run migrations, check sandbox |
| `uv run ash sandbox build` | Build sandbox image |
| `uv run ash memory <action>` | Manage memories (list, search, add, remove, gc, stats) |
| `uv run ash sessions <action>` | Manage sessions (list, search, export, clear) |

## Conventions

- Async everywhere (`async def`, `await`)
- Type hints required
- Pydantic for validation
- ABC for interfaces in `*/base.py`
- Tests in `tests/`

## Logging

| Level | Use For |
|-------|---------|
| DEBUG | Development tracing, cache hits, API internals |
| INFO | User-visible operations, tool/skill summaries |
| WARNING | Recoverable issues, retries, missing optional config |
| ERROR | Failures that affect operation |

**Rules:**
1. Single source of truth - each operation logged in one place only
2. Tools: logged in `executor.py` only (with timing)
3. LLM API calls: DEBUG level (too noisy for INFO)
4. Use `ASH_LOG_LEVEL=DEBUG` for development

**Configuration:**
- All entry points use `ash.logging.configure_logging()`
- Server mode: Rich formatting with `use_rich=True`
- Chat mode: Suppressed to WARNING (TUI controls display)

## Commit Attribution

AI commits MUST include:
```
Co-Authored-By: (the agent model's name and attribution byline)
```

## Specifications

Use `/write-spec` skill for new features. See `.claude/skills/write-spec.md`

Use `/verify-spec` skill to check implementation. See `.claude/skills/verify-spec.md`

- Specs live in `specs/<feature>.md`
- Update spec BEFORE implementing
- Format defined in `SPECS.md`

## Database

Use `/create-migration` skill for schema changes. See `.claude/skills/create-migration.md`

- Run migrations: `uv run alembic upgrade head`
- Check status: `uv run alembic current`

## Documentation

Use `/write-docs` skill for docs site pages. See `.claude/skills/write-docs.md`

- Docs site: `docs/` (Astro Starlight)
- Run locally: `cd docs && pnpm dev`

## Other Skills

| Skill | When to use |
|-------|-------------|
| `/commit` | Creating commits |
| `/create-pr` | Opening pull requests |
| `/find-bugs` | Pre-merge review |

## Reference

- `SPECS.md` - Spec format and index
- `ARCHITECTURE.md` - Tech stack and roadmap
