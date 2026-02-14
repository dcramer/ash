"""Plan agent behavior evaluation tests.

Tests the plan agent's ability to create structured, scannable implementation
plans and checkpoint for user approval.

Run with: uv run pytest evals/test_plan_agent.py -v -s
Requires ANTHROPIC_API_KEY environment variable.
"""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from ash.agents.base import AgentContext
from ash.agents.builtin.plan import PlanAgent
from ash.agents.executor import AgentExecutor
from ash.config import AshConfig
from ash.core.agent import AgentComponents
from ash.llm.base import LLMProvider
from evals.judge import LLMJudge
from evals.runner import get_case_by_id, load_eval_suite
from evals.types import EvalCase

CASES_DIR = Path(__file__).parent / "cases"
PLAN_AGENT_CASES = CASES_DIR / "plan_agent.yaml"


@pytest.fixture
async def plan_agent_executor(
    eval_config: AshConfig,
    eval_agent: AgentComponents,
) -> AsyncGenerator[tuple[PlanAgent, AgentExecutor], None]:
    """Create plan agent with executor for testing."""
    agent = PlanAgent()
    executor = AgentExecutor(
        llm_provider=eval_agent.llm,
        tool_executor=eval_agent.tool_executor,
        config=eval_config,
    )
    yield agent, executor


@pytest.mark.eval
class TestPlanAgent:
    """Core plan agent behavior tests."""

    @pytest.mark.asyncio
    async def test_produces_structured_plan(
        self,
        plan_agent_executor: tuple[PlanAgent, AgentExecutor],
        judge_llm: LLMProvider,
    ) -> None:
        """Plan agent should produce a structured, scannable plan via checkpoint."""
        agent, executor = plan_agent_executor
        suite = load_eval_suite(PLAN_AGENT_CASES)
        case = get_case_by_id(suite, "structured_plan")

        context = AgentContext(
            session_id="test-plan",
            chat_id="test-chat",
            user_id="test-user",
        )

        result = await executor.execute(agent, case.prompt, context)

        # Must checkpoint with a plan
        assert result.checkpoint is not None, "Plan agent should checkpoint"
        assert result.checkpoint.options, "Checkpoint should provide options"

        plan_text = result.checkpoint.prompt

        # Structural checks (deterministic)
        lines = plan_text.split("\n")
        bullet_lines = [
            line
            for line in lines
            if line.strip().startswith(("-", "*", "1.", "2.", "3."))
        ]
        assert len(bullet_lines) >= 3, "Plan should use bullet points"

        non_empty_lines = [line for line in lines if line.strip()]
        assert len(non_empty_lines) <= 100, (
            f"Plan too verbose: {len(non_empty_lines)} lines"
        )

        # Quality check via judge
        judge_case = EvalCase(
            id="plan_quality",
            description="Evaluate plan structure and content",
            prompt=case.prompt,
            expected_behavior="""
                A good implementation plan should:
                1. Have a clear approach summary
                2. Be organized into phases
                3. Each phase should have specific actions
                4. Include verification steps
            """,
            criteria=[
                "Has approach/summary section",
                "Has multiple phases",
                "Uses bullet points for scannability",
                "Includes verification or test steps",
            ],
        )

        judge = LLMJudge(judge_llm)
        judge_result = await judge.evaluate(
            case=judge_case,
            response_text=plan_text,
            tool_calls=[],
        )

        print(f"\nPlan checkpoint:\n{plan_text}")
        print(f"\nJudge passed: {judge_result.passed}")
        print(f"Judge reasoning: {judge_result.reasoning}")

        if judge_result.judge_error:
            pytest.skip(f"Judge error: {judge_result.reasoning}")

        assert judge_result.passed, (
            f"Plan quality check failed: {judge_result.reasoning}"
        )
