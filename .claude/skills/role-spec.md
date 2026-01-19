# /role-spec

Spec quality, completeness, and implementation alignment.

## Responsibility

The role-spec owns specification quality. It:
- Audits specs for completeness
- Checks requirements have corresponding tests
- Identifies stale or outdated specs
- Ensures implementation matches spec

## Tools

```bash
# Spec audit (when available)
uv run python scripts/spec-audit.py

# List all specs
ls specs/

# Verify spec against implementation
# Use /verify-spec skill for detailed check
```

## Process

1. **Inventory specs**
   ```bash
   ls specs/
   ```
   List all specification files.

2. **Audit each spec**
   For each spec file:
   - Are all MUST requirements testable?
   - Do MUST requirements have tests?
   - Is the spec current with implementation?
   - Are the interfaces documented?

3. **Check spec format**
   Each spec should have:
   ```markdown
   # Feature Name

   > One-line purpose

   Files: path/to/file.py

   ## Requirements
   ### MUST
   ### SHOULD
   ### MAY

   ## Interface
   ## Behaviors
   ## Errors
   ## Verification
   ```

4. **Identify gaps**
   - Missing specs for features
   - Unverified requirements
   - Implementation drift

5. **Recommend updates**
   - Specific requirements to add
   - Tests to write
   - Sections to update

## Handoff

**Receiving work:**
- From role-master: spec audit request
- Before implementation: clarify requirements

**Reporting results:**
- Specs audited
- Issues found (missing tests, stale content)
- Recommendations

## Spec Quality Checklist

- [ ] One-line purpose statement
- [ ] Files listed
- [ ] MUST requirements are testable
- [ ] SHOULD/MAY clearly distinguished
- [ ] Interface documented
- [ ] Behaviors table complete
- [ ] Error conditions listed
- [ ] Verification commands work

## Using /verify-spec

For detailed implementation verification:
```
/verify-spec <feature>
```

This checks:
- All MUST requirements implemented
- Tests exist for requirements
- Implementation matches interface

## Rules

- Every feature should have a spec
- Every MUST requirement needs a test
- Specs are stateless (no implementation tracking)
- No "why" explanations in specs (put in docs)
- Specs describe what, not how
- Update spec before implementation changes
