# Docs Pages

> User-focused structure and verification rules for docs pages

Files: docs/src/content/docs/systems/*.mdx

## Requirements

### MUST

- Lead with a concise problem-oriented intro (1-3 paragraphs, no implementation details).
- Include a quickstart section with runnable commands or concrete first actions.
- Use action-first headings that map to user tasks.
- Use commented code blocks for option explanations instead of duplicate code-plus-table explanations.
- Keep one canonical path recommendation per workflow and avoid contradictory path guidance.
- Include a troubleshooting section with concrete commands and outcomes.
- Move implementation internals to a final `Reference (Advanced)` section.
- Use Plain Speech tone for documentation copy.
- Keep callouts short and task-relevant.

### SHOULD

- Use consistent section order across systems pages:
  1. `X in 30 Seconds` or equivalent quick orientation
  2. `Where/When` guidance
  3. `Quick Start` or `Create/Setup`
  4. `Troubleshooting`
  5. `Reference (Advanced)`
- Prefer one continuous example per page.
- Keep command blocks grouped by workflow (setup, validate, diagnose).
- Include configuration snippets only where they are directly relevant to the page.

### MAY

- Include optional compatibility notes in advanced/reference sections.
- Include architecture links to related specs in advanced/reference sections.

## Interface

### Canonical Page Skeleton

````markdown
# <Topic>

> One-line user benefit statement

## <Topic> in 30 Seconds
- What it is
- When to use it

## Where <Thing> Should Live
- Canonical path or location guidance

## Quick Start
~~~bash
# minimal first-success workflow
~~~

## Troubleshooting
### <Symptom>
~~~bash
# verify and fix
~~~

## Reference (Advanced)
- internals, storage layout, implementation notes
````

### Content Rules

```text
- Do not repeat the same option explanation in both a table and a code block.
- Prefer short bulleted steps over long prose paragraphs.
- Use American English.
- Keep labels/headings in Title Case; body and commands in sentence case.
```

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|
| Page contains option table and equivalent commented config block | Keep commented config block, remove duplicate table | Applies to settings/options sections |
| Page has deep internals before setup guidance | Move internals to `Reference (Advanced)` | Keep main flow user-task-first |
| Page has multiple path variants without guidance | Keep one canonical path and label alternates as runtime/internal mapping | Avoid user ambiguity |
| Page has troubleshooting prose without commands | Add runnable verification/fix commands | Troubleshooting must be actionable |

## Errors

| Condition | Response |
|-----------|----------|
| Contradictory path guidance on a page | Fail docs review; consolidate to a canonical path and one mapping note |
| Duplicate explanation via table and code block | Fail docs review; keep one source of truth |
| Missing quickstart/first-success path | Fail docs review; add minimal runnable workflow |
| Internals mixed into primary user flow | Fail docs review; move to `Reference (Advanced)` |

## Verification

```bash
# Ensure systems pages have a troubleshooting section
for f in docs/src/content/docs/systems/*.mdx; do rg -q "^## Troubleshooting" "$f" || echo "missing troubleshooting: $f"; done

# Ensure systems pages carry advanced/internal section for deep details
for f in docs/src/content/docs/systems/*.mdx; do rg -q "^## Reference \(Advanced\)" "$f" || echo "missing advanced reference: $f"; done

# Find table-heavy sections to review for possible code-comment replacement
for f in docs/src/content/docs/systems/*.mdx; do rg -n "^\|.*\|" "$f" | head -n 20; done

# Find config/code blocks and verify options are documented inline where applicable
for f in docs/src/content/docs/systems/*.mdx; do rg -n '```toml|```yaml|```bash' "$f" | head -n 20; done
```

- Verify each systems page has a first-success workflow that can be followed without reading advanced sections.
- Verify each options section uses one explanation mode (commented code block preferred).
- Verify heading language is task-oriented and user-facing.
