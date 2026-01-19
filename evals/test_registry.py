"""Agent registry behavior evaluation tests.

Tests the use_agent tool and agent delegation system:
- Invoking registered agents (research, plan, skill_writer)
- Handling non-existent agent requests
- Passing context to delegated agents

Run with: uv run pytest evals/test_registry.py -v
Requires ANTHROPIC_API_KEY environment variable.
"""

from pathlib import Path

import pytest

from ash.core.agent import AgentComponents
from ash.llm.base import LLMProvider
from evals.report import print_report
from evals.runner import get_case_by_id, load_eval_suite, run_eval_case, run_eval_suite

CASES_DIR = Path(__file__).parent / "cases"
REGISTRY_CASES = CASES_DIR / "registry.yaml"

# Default accuracy threshold (can be overridden per-suite)
DEFAULT_ACCURACY_THRESHOLD = 0.80


def _get_accuracy_threshold(suite_path: Path) -> float:
    """Get accuracy threshold for a suite.

    Uses suite-specific threshold if defined, otherwise default.
    """
    suite = load_eval_suite(suite_path)
    if suite.accuracy_threshold is not None:
        return suite.accuracy_threshold
    return DEFAULT_ACCURACY_THRESHOLD


@pytest.mark.eval
class TestRegistryEvals:
    """Agent registry behavior evaluation tests."""

    @pytest.mark.asyncio
    async def test_invoke_research_agent(
        self,
        eval_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """Test invoking research agent via use_agent tool."""
        suite = load_eval_suite(REGISTRY_CASES)
        case = get_case_by_id(suite, "invoke_research_agent")

        result = await run_eval_case(
            agent=eval_agent.agent,
            case=case,
            judge_llm=judge_llm,
        )

        # Print detailed result for debugging
        print(f"\nResponse: {result.response_text[:200]}...")
        print(f"Score: {result.score:.2f}")
        print(f"Reasoning: {result.judge_result.reasoning}")

        if result.is_judge_error:
            pytest.fail(f"Judge error: {result.judge_result.reasoning}")
        assert result.passed, f"Eval failed: {result.judge_result.reasoning}"

    @pytest.mark.asyncio
    async def test_invoke_nonexistent_agent(
        self,
        eval_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """Test handling of non-existent agent invocation."""
        suite = load_eval_suite(REGISTRY_CASES)
        case = get_case_by_id(suite, "invoke_nonexistent_agent")

        result = await run_eval_case(
            agent=eval_agent.agent,
            case=case,
            judge_llm=judge_llm,
        )

        print(f"\nResponse: {result.response_text[:200]}...")
        print(f"Score: {result.score:.2f}")
        print(f"Reasoning: {result.judge_result.reasoning}")

        if result.is_judge_error:
            pytest.fail(f"Judge error: {result.judge_result.reasoning}")
        assert result.passed, f"Eval failed: {result.judge_result.reasoning}"

    @pytest.mark.asyncio
    async def test_agent_delegation_with_context(
        self,
        eval_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """Test delegating to plan agent with context."""
        suite = load_eval_suite(REGISTRY_CASES)
        case = get_case_by_id(suite, "agent_delegation_with_context")

        result = await run_eval_case(
            agent=eval_agent.agent,
            case=case,
            judge_llm=judge_llm,
        )

        print(f"\nResponse: {result.response_text[:200]}...")
        print(f"Score: {result.score:.2f}")
        print(f"Reasoning: {result.judge_result.reasoning}")

        if result.is_judge_error:
            pytest.fail(f"Judge error: {result.judge_result.reasoning}")
        assert result.passed, f"Eval failed: {result.judge_result.reasoning}"

    @pytest.mark.asyncio
    async def test_list_available_agents(
        self,
        eval_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """Test that assistant can describe available agents."""
        suite = load_eval_suite(REGISTRY_CASES)
        case = get_case_by_id(suite, "list_available_agents")

        result = await run_eval_case(
            agent=eval_agent.agent,
            case=case,
            judge_llm=judge_llm,
        )

        print(f"\nResponse: {result.response_text[:200]}...")
        print(f"Score: {result.score:.2f}")
        print(f"Reasoning: {result.judge_result.reasoning}")

        if result.is_judge_error:
            pytest.fail(f"Judge error: {result.judge_result.reasoning}")
        assert result.passed, f"Eval failed: {result.judge_result.reasoning}"

    @pytest.mark.asyncio
    async def test_full_registry_suite(
        self,
        eval_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """Run all registry eval cases and assert accuracy threshold."""
        suite = load_eval_suite(REGISTRY_CASES)
        threshold = _get_accuracy_threshold(REGISTRY_CASES)

        report = await run_eval_suite(
            agent=eval_agent.agent,
            suite=suite,
            judge_llm=judge_llm,
        )

        # Print rich report
        print_report(report)

        # Warn about judge errors
        if report.judge_errors > 0:
            print(
                f"\nWarning: {report.judge_errors} cases had judge errors and were excluded from accuracy"
            )

        # Assert accuracy threshold
        assert report.accuracy >= threshold, (
            f"Accuracy {report.accuracy:.2%} below threshold {threshold:.2%}. "
            f"Failed cases: {[r.case.id for r in report.failed_cases()]}"
        )
