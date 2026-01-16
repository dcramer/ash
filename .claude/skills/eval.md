# /eval

Run or write evaluation tests for agent behavior.

## Usage

```
/eval run [suite]     # Run eval suite (default: all)
/eval write <feature> # Write new eval cases for a feature
```

## Running Evals

Run evals with real LLM calls to test agent behavior:

```bash
# Run all evals
uv run pytest evals -m eval -v

# Run specific suite
uv run pytest evals/test_scheduler.py -v -s

# Run single test
uv run pytest evals/test_scheduler.py::TestSchedulerEvals::test_schedule_simple_reminder -v -s
```

Requires `ANTHROPIC_API_KEY` environment variable.

## Writing Evals

### 1. Create YAML case file

Location: `evals/cases/<feature>.yaml`

```yaml
name: Feature Name
description: What this suite tests

cases:
  - id: unique_case_id
    description: Human-readable description
    prompt: "User message to send to agent"
    expected_behavior: |
      What the agent should do in response.
      Be specific about expected actions and outputs.
    criteria:
      - Specific criterion 1 the judge evaluates
      - Specific criterion 2
    expected_tools:
      - tool_name  # Tools that should be called (optional)
    tags:
      - feature_tag
```

### 2. Create test file

Location: `evals/test_<feature>.py`

```python
"""Feature behavior evaluation tests."""

from pathlib import Path
import pytest
from ash.core.agent import Agent
from ash.llm.base import LLMProvider
from evals.report import print_report
from evals.runner import load_eval_suite, run_eval_case, run_eval_suite

CASES_DIR = Path(__file__).parent / "cases"
FEATURE_CASES = CASES_DIR / "feature.yaml"
ACCURACY_THRESHOLD = 0.80

@pytest.mark.eval
class TestFeatureEvals:
    @pytest.mark.asyncio
    async def test_specific_case(
        self,
        eval_agent: Agent,
        judge_llm: LLMProvider,
    ) -> None:
        suite = load_eval_suite(FEATURE_CASES)
        case = next(c for c in suite.cases if c.id == "specific_case_id")
        result = await run_eval_case(agent=eval_agent, case=case, judge_llm=judge_llm)
        assert result.passed, f"Eval failed: {result.judge_result.reasoning}"

    @pytest.mark.asyncio
    async def test_full_suite(
        self,
        eval_agent: Agent,
        judge_llm: LLMProvider,
    ) -> None:
        suite = load_eval_suite(FEATURE_CASES)
        report = await run_eval_suite(agent=eval_agent, suite=suite, judge_llm=judge_llm)
        print_report(report)
        assert report.accuracy >= ACCURACY_THRESHOLD
```

## Eval Framework Components

- `evals/types.py` - Pydantic models: `EvalCase`, `EvalSuite`, `JudgeResult`
- `evals/judge.py` - LLM-as-judge evaluation
- `evals/runner.py` - `load_eval_suite()`, `run_eval_case()`, `run_eval_suite()`
- `evals/report.py` - Rich terminal output with `print_report()`
- `evals/conftest.py` - Fixtures: `eval_agent`, `judge_llm`, `real_llm`

## Writing Good Eval Cases

1. **Clear prompts**: Write prompts as a real user would
2. **Specific expected behavior**: Describe what success looks like
3. **Measurable criteria**: Each criterion should be binary (met/not met)
4. **Independent cases**: Each case should test one behavior
5. **Appropriate threshold**: Set accuracy threshold based on case difficulty

## Rules

- Evals use real LLM calls (cost money)
- Judge model: `claude-sonnet-4-5` (reliable evaluation)
- Agent model: `claude-sonnet-4-5` (consistent behavior)
- Never hardcode dated model names (use `claude-sonnet-4-5` not `claude-sonnet-4-5-20250514`)
- Fresh session per eval case (isolation)
- Threshold-based assertions (80% default, adjust per suite)
