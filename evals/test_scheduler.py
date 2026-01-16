"""Scheduler behavior evaluation tests.

These tests use real LLM calls to evaluate agent behavior.
Run with: uv run pytest evals/test_scheduler.py -v

Requires ANTHROPIC_API_KEY environment variable.
"""

from pathlib import Path

import pytest

from ash.core.agent import AgentComponents
from ash.llm.base import LLMProvider
from evals.report import print_report
from evals.runner import load_eval_suite, run_eval_case, run_eval_suite

CASES_DIR = Path(__file__).parent / "cases"
SCHEDULER_CASES = CASES_DIR / "scheduler.yaml"

# Minimum accuracy threshold for the full suite
ACCURACY_THRESHOLD = 0.80


@pytest.mark.eval
class TestSchedulerEvals:
    """Scheduler behavior evaluation tests."""

    @pytest.mark.asyncio
    async def test_schedule_simple_reminder(
        self,
        eval_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """Test simple reminder scheduling."""
        suite = load_eval_suite(SCHEDULER_CASES)
        case = next(c for c in suite.cases if c.id == "schedule_simple_reminder")

        result = await run_eval_case(
            agent=eval_agent.agent,
            case=case,
            judge_llm=judge_llm,
        )

        # Print detailed result for debugging
        print(f"\nResponse: {result.response_text[:200]}...")
        print(f"Score: {result.score:.2f}")
        print(f"Reasoning: {result.judge_result.reasoning}")

        assert result.passed, f"Eval failed: {result.judge_result.reasoning}"

    @pytest.mark.asyncio
    async def test_schedule_with_timezone(
        self,
        eval_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """Test timezone-aware scheduling."""
        suite = load_eval_suite(SCHEDULER_CASES)
        case = next(c for c in suite.cases if c.id == "schedule_with_timezone")

        result = await run_eval_case(
            agent=eval_agent.agent,
            case=case,
            judge_llm=judge_llm,
        )

        print(f"\nResponse: {result.response_text[:200]}...")
        print(f"Score: {result.score:.2f}")
        print(f"Reasoning: {result.judge_result.reasoning}")

        assert result.passed, f"Eval failed: {result.judge_result.reasoning}"

    @pytest.mark.asyncio
    async def test_schedule_recurring(
        self,
        eval_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """Test recurring reminder handling."""
        suite = load_eval_suite(SCHEDULER_CASES)
        case = next(c for c in suite.cases if c.id == "schedule_recurring")

        result = await run_eval_case(
            agent=eval_agent.agent,
            case=case,
            judge_llm=judge_llm,
        )

        print(f"\nResponse: {result.response_text[:200]}...")
        print(f"Score: {result.score:.2f}")
        print(f"Reasoning: {result.judge_result.reasoning}")

        assert result.passed, f"Eval failed: {result.judge_result.reasoning}"

    @pytest.mark.asyncio
    async def test_schedule_vague_time(
        self,
        eval_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """Test handling of vague time references."""
        suite = load_eval_suite(SCHEDULER_CASES)
        case = next(c for c in suite.cases if c.id == "schedule_vague_time")

        result = await run_eval_case(
            agent=eval_agent.agent,
            case=case,
            judge_llm=judge_llm,
        )

        print(f"\nResponse: {result.response_text[:200]}...")
        print(f"Score: {result.score:.2f}")
        print(f"Reasoning: {result.judge_result.reasoning}")

        assert result.passed, f"Eval failed: {result.judge_result.reasoning}"

    @pytest.mark.asyncio
    async def test_full_scheduler_suite(
        self,
        eval_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """Run all scheduler eval cases and assert accuracy threshold."""
        suite = load_eval_suite(SCHEDULER_CASES)

        report = await run_eval_suite(
            agent=eval_agent.agent,
            suite=suite,
            judge_llm=judge_llm,
        )

        # Print rich report
        print_report(report)

        # Assert accuracy threshold
        assert report.accuracy >= ACCURACY_THRESHOLD, (
            f"Accuracy {report.accuracy:.2%} below threshold {ACCURACY_THRESHOLD:.2%}. "
            f"Failed cases: {[r.case.id for r in report.failed_cases()]}"
        )
