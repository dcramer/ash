"""Agent registry behavior evaluation tests.

Tests agent delegation chains (research → plan) and error recovery
(nonexistent agent → fallback) as multi-turn conversations.

Run with: uv run pytest evals/test_registry.py -v -s -m eval
Requires ANTHROPIC_API_KEY environment variable.
"""

import logging
from pathlib import Path

import pytest

from ash.core.agent import AgentComponents
from ash.core.session import SessionState
from ash.llm.base import LLMProvider
from evals.judge import LLMJudge
from evals.runner import get_case_by_id, load_eval_suite
from evals.types import EvalConfig

logger = logging.getLogger(__name__)

CASES_DIR = Path(__file__).parent / "cases"
REGISTRY_CASES = CASES_DIR / "registry.yaml"


def _make_session(session_id: str) -> SessionState:
    """Create an eval session for registry tests."""
    return SessionState(
        session_id=session_id,
        provider="eval",
        chat_id="eval-registry-chat",
        user_id="eval-user",
    )


@pytest.mark.eval
class TestRegistryEvals:
    """Agent registry behavior evaluation tests."""

    @pytest.mark.asyncio
    async def test_registry_delegation_chain(
        self,
        eval_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """Research → plan delegation: research agent output feeds into plan creation."""
        agent = eval_agent.agent
        session = _make_session("eval-registry-delegation")

        suite = load_eval_suite(REGISTRY_CASES)
        case = get_case_by_id(suite, "registry_delegation_chain")

        response = await agent.process_message(
            case.prompt,
            session=session,
            user_id="eval-user",
        )

        judge = LLMJudge(judge_llm, EvalConfig())
        result = await judge.evaluate(
            case=case,
            response_text=response.text,
            tool_calls=response.tool_calls,
        )

        logger.info("[%s] Response: %s", case.id, response.text)
        logger.info(
            "[%s] Judge: passed=%s, score=%s", case.id, result.passed, result.score
        )
        logger.info("[%s] Reasoning: %s", case.id, result.reasoning)
        if result.criteria_scores:
            logger.info("[%s] Criteria: %s", case.id, result.criteria_scores)

        assert result.passed, (
            f"{case.id} failed (score={result.score}): {result.reasoning}"
        )

    @pytest.mark.asyncio
    async def test_registry_error_recovery(
        self,
        eval_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """Nonexistent agent → fallback: recover from foobar agent to research agent."""
        agent = eval_agent.agent
        session = _make_session("eval-registry-error")

        suite = load_eval_suite(REGISTRY_CASES)
        case = get_case_by_id(suite, "registry_error_recovery")

        response = await agent.process_message(
            case.prompt,
            session=session,
            user_id="eval-user",
        )

        judge = LLMJudge(judge_llm, EvalConfig())
        result = await judge.evaluate(
            case=case,
            response_text=response.text,
            tool_calls=response.tool_calls,
        )

        logger.info("[%s] Response: %s", case.id, response.text)
        logger.info(
            "[%s] Judge: passed=%s, score=%s", case.id, result.passed, result.score
        )
        logger.info("[%s] Reasoning: %s", case.id, result.reasoning)
        if result.criteria_scores:
            logger.info("[%s] Criteria: %s", case.id, result.criteria_scores)

        assert result.passed, (
            f"{case.id} failed (score={result.score}): {result.reasoning}"
        )
