"""Scheduler behavior evaluation tests.

Tests the scheduling system via suite-level accuracy threshold.
Individual cases are defined in cases/scheduler.yaml.

Run with: uv run pytest evals/test_scheduler.py -v
Requires ANTHROPIC_API_KEY environment variable.
"""

from pathlib import Path

import pytest

from ash.core.agent import AgentComponents
from ash.llm.base import LLMProvider
from evals.report import print_report
from evals.runner import load_eval_suite, run_eval_suite

CASES_DIR = Path(__file__).parent / "cases"
SCHEDULER_CASES = CASES_DIR / "scheduler.yaml"

DEFAULT_ACCURACY_THRESHOLD = 0.80


@pytest.mark.eval
class TestSchedulerEvals:
    """Scheduler behavior evaluation tests."""

    @pytest.mark.asyncio
    async def test_full_scheduler_suite(
        self,
        eval_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """Run all scheduler eval cases and assert accuracy threshold."""
        suite = load_eval_suite(SCHEDULER_CASES)
        threshold = suite.accuracy_threshold or DEFAULT_ACCURACY_THRESHOLD

        report = await run_eval_suite(
            agent=eval_agent.agent,
            suite=suite,
            judge_llm=judge_llm,
        )

        print_report(report)

        if report.judge_errors > 0:
            print(
                f"\nWarning: {report.judge_errors} cases had judge errors and were excluded from accuracy"
            )

        assert report.accuracy >= threshold, (
            f"Accuracy {report.accuracy:.2%} below threshold {threshold:.2%}. "
            f"Failed cases: {[r.case.id for r in report.failed_cases()]}"
        )
