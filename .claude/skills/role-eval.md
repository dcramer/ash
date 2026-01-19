# /role-eval

Write evals, analyze judge quality, and track behavior coverage.

## Responsibility

The role-eval owns behavior evaluation. It:
- Identifies behaviors that need eval coverage
- Writes eval cases to test agent behavior
- Analyzes judge consistency
- Tracks eval pass rates

## Tools

```bash
# Eval coverage analysis
uv run python scripts/eval-coverage.py

# JSON output
uv run python scripts/eval-coverage.py --json

# Run all evals
uv run pytest evals -m eval -v

# Run specific suite
uv run pytest evals/test_scheduler.py -v -s

# Run single eval case
uv run pytest evals/test_scheduler.py::TestSchedulerEvals::test_schedule_simple_reminder -v -s
```

## Process

1. **Analyze coverage**
   ```bash
   uv run python scripts/eval-coverage.py
   ```
   Note:
   - Agents without eval suites
   - Behaviors without coverage
   - Existing suite quality

2. **Identify behaviors to test**
   For an agent or feature:
   - What are the core capabilities?
   - What edge cases matter?
   - What failure modes should be tested?

3. **Write eval cases**
   Create/update YAML case file: `evals/cases/<feature>.yaml`

   ```yaml
   name: Feature Name
   description: What this suite tests

   cases:
     - id: unique_case_id
       description: Human-readable description
       prompt: "User message to send to agent"
       expected_behavior: |
         What the agent should do in response.
         Be specific about expected actions.
       criteria:
         - Specific criterion the judge evaluates
         - Another specific criterion
       expected_tools:
         - tool_name
       tags:
         - feature_tag
   ```

4. **Create test file if needed**
   Location: `evals/test_<feature>.py`

   Follow the pattern in `/eval` skill for test structure.

5. **Run evals**
   ```bash
   uv run pytest evals/test_<feature>.py -v -s
   ```
   Note:
   - Pass rate
   - Judge reasoning for failures
   - Consistency across runs

6. **Report results**
   - Cases added/modified
   - Pass rate
   - Coverage improvement
   - Any judge inconsistencies

## Handoff

**Receiving work:**
- From role-master: feature/behavior to add eval coverage
- From role-qa: areas where unit tests aren't sufficient

**Reporting results:**
- Eval suite created/modified
- Pass rate (target: 80%+)
- Coverage improvement
- Judge quality notes

## Writing Good Evals

1. **Clear prompts**: Write as a real user would
2. **Specific expected behavior**: Describe success precisely
3. **Measurable criteria**: Binary (met/not met)
4. **Independent cases**: One behavior per case
5. **Appropriate threshold**: 80% default, adjust per difficulty

## Rules

- Evals use real LLM calls (cost money)
- Use `claude-sonnet-4-5` for judge model
- Fresh session per eval case
- Threshold: 80% accuracy minimum
- Never hardcode dated model names
- Each case should test one specific behavior
