# /verify-spec

Verify implementation matches a feature specification.

## Usage

```
/verify-spec <feature>
```

## Process

1. Read spec: `specs/<feature>.md`
2. Run verification commands from spec
3. Check each requirement:
   - MUST requirements
   - SHOULD requirements
   - MAY requirements
4. Report result:
   - **PASS**: All MUST + SHOULD requirements met
   - **PARTIAL**: All MUST requirements met, some SHOULD missing
   - **FAIL**: Missing MUST requirements

## Output Format

```
## <Feature> Spec Verification

**Result: PASS | PARTIAL | FAIL**

### Tests
- Test results summary

### MUST Requirements
| Requirement | Status | Evidence |
|-------------|--------|----------|
| ... | PASS/FAIL | ... |

### SHOULD Requirements
| Requirement | Status | Evidence |
|-------------|--------|----------|
| ... | PASS/FAIL | ... |

### MAY Requirements
| Requirement | Status |
|-------------|--------|
| ... | Implemented/Not implemented |

### Verification Checklist
- [x] Passing checks
- [ ] Failing checks
```

## Rules

- Run all verification commands from the spec
- Check implementation files exist
- Verify interfaces match spec signatures
- Test behaviors match expected outputs
- Document evidence for each requirement
