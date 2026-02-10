---
name: architecture-review
description: Use when reviewing codebase health, finding structural issues, or auditing architecture quality before major changes
---

# /architecture-review

Staff-level codebase health review. Focus on macro-level structural issues that compound over time, not style preferences.

## Five Review Dimensions

### 1. Module Complexity

- **Size**: Flag files >500 lines; investigate >800 as monoliths
- **Responsibilities**: >3 distinct concerns (validation, I/O, orchestration) signals splitting
- **Fan-out**: Files importing 10+ modules may coordinate excessively

For oversized modules, propose specific splits with new file names.

### 2. Silent Failure Patterns

- Swallowed exceptions returning defaults without logging
- Empty returns (`[]`, `None`) where callers can't distinguish "no results" from "failed"
- Missing error callbacks on async operations
- Silent fallbacks hiding upstream problems

Explain what information gets lost and how to surface it.

### 3. Type Safety Gaps

- `cast()` or `# type: ignore` without runtime validation
- `Optional` returns where `None` means both "not found" and "error"
- Overly broad exception handling (`except Exception`)
- Missing Pydantic validation at boundaries

Suggest type-safe alternatives.

### 4. Test Coverage Analysis

- Untested critical paths (core logic, error handling)
- Edge case gaps (empty inputs, boundaries)
- Integration gaps where only unit tests exist
- Bug fixes lacking corresponding tests

Prioritize by risk: untested hot paths > edge cases > utilities.

### 5. LLM-Friendliness

- Docstring coverage on public functions
- Clear naming without needing implementation context
- Actionable error messages explaining problems and fixes
- Configuration that's easy to misconfigure

## Analysis Method

1. **Map architecture**: Read entry points, understand module structure
   ```bash
   cat specs/subsystems.md
   ```

2. **Find giants**: Locate largest files
   ```bash
   find src -name "*.py" -exec wc -l {} + | sort -rn | head -20
   ```

3. **Trace error paths**: Follow operation failures, identify where error info disappears

4. **Audit type assertions**: Search for `cast()`, `# type: ignore`, broad exceptions
   ```bash
   rg "cast\(|type: ignore|except Exception" src/
   ```

5. **Map test coverage**: Compare test files against source
   ```bash
   ls tests/ && ls src/ash/
   ```

6. **Check subsystem boundaries**: Verify import rules from `specs/subsystems.md`

## Output Format

**Executive Summary**: 3-5 bullets on most impactful findings

**Priority Issues**: For each finding:
- Problem: What's wrong and why it matters
- Evidence: Specific files, line numbers, patterns
- Recommendation: Concrete fix

**What's Working Well**: Architectural strengths to preserve

## Severity Levels

- **critical**: Causing active reliability problems
- **high**: Compounding as codebase grows
- **medium**: Worth fixing but not urgent
- **low**: Nice-to-have

## Exclusions

Do NOT report: style preferences, minor naming issues, single-line fixes, already-addressed issues.
