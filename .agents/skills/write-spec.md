# /write-spec

Create or update a feature specification.

## Usage

```
/write-spec <feature>
```

## Process

1. Read project context: `CLAUDE.md`, `ARCHITECTURE.md`, `SPECS.md`
2. Read existing spec if present: `specs/<feature>.md`
3. Read implementation files to understand current state
4. Draft spec with: requirements, interface, behaviors, errors, verification
5. Review against project goals:
   - Does this spec serve the project's purpose?
   - Does it integrate properly with other features?
6. Revise if the spec doesn't align with project objectives
7. Follow format in `SPECS.md`
8. Update `SPECS.md` index if new spec

## Spec Format

Follow the exact format from `SPECS.md`:

```markdown
# Feature Name

> One-line purpose statement

Files: path/to/file.py, path/to/other.py

## Requirements

### MUST
- Requirement with testable criteria

### SHOULD
- Nice-to-have with testable criteria

### MAY
- Optional behavior

## Interface

```python
def function(param: Type) -> ReturnType: ...
```

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|

## Errors

| Condition | Response |
|-----------|----------|

## Verification

```bash
command_to_test_feature
```

- Verification check 1
- Verification check 2
```

## Rules

- Every requirement must be testable
- No design rationale or "why" explanations
- No implementation hints
- No verbose prose
- Specs are stateless - no tracking of implementation status
