"""Scheduler behavior evaluation tests.

Multi-turn scheduling test: create reminders (recurring + one-off),
list them, then cancel one — verifying the full scheduling lifecycle
in a single conversation flow.

Run with: uv run pytest evals/test_scheduler.py -v -s -m eval
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
from evals.types import EvalCase, EvalConfig

logger = logging.getLogger(__name__)

CASES_DIR = Path(__file__).parent / "cases"
SCHEDULER_CASES = CASES_DIR / "scheduler.yaml"


def _make_session(session_id: str) -> SessionState:
    """Create an eval session for scheduler tests."""
    return SessionState(
        session_id=session_id,
        provider="eval",
        chat_id="eval-scheduler-chat",
        user_id="eval-user",
    )


async def _judge_response(
    judge_llm: LLMProvider,
    case: EvalCase,
    response_text: str,
    tool_calls: list,
) -> None:
    """Judge a response and assert it passes."""
    judge = LLMJudge(judge_llm, EvalConfig())
    result = await judge.evaluate(
        case=case,
        response_text=response_text,
        tool_calls=tool_calls,
    )

    logger.info("[%s] Response: %s", case.id, response_text)
    logger.info("[%s] Judge: passed=%s, score=%s", case.id, result.passed, result.score)
    logger.info("[%s] Reasoning: %s", case.id, result.reasoning)
    if result.criteria_scores:
        logger.info("[%s] Criteria: %s", case.id, result.criteria_scores)

    assert result.passed, f"{case.id} failed (score={result.score}): {result.reasoning}"


@pytest.mark.eval
class TestSchedulerEvals:
    """Multi-turn scheduler behavior evaluation."""

    @pytest.mark.asyncio
    async def test_scheduler_multi_step(
        self,
        eval_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        """Multi-turn: create reminders, list them, cancel one.

        Turn 1: Create recurring standup + one-off dentist reminder
        Turn 2: "What reminders do I have?" — should list both
        Turn 3: "Cancel the dentist one" — should cancel only the dentist
        """
        agent = eval_agent.agent
        session = _make_session("eval-scheduler-multi")

        # --- Turn 1: Create both reminders ---
        suite = load_eval_suite(SCHEDULER_CASES)
        create_case = get_case_by_id(suite, "scheduler_multi_step")

        response1 = await agent.process_message(
            create_case.prompt,
            session=session,
            user_id="eval-user",
        )

        await _judge_response(
            judge_llm, create_case, response1.text, response1.tool_calls
        )

        # --- Turn 2: List reminders ---
        list_case = EvalCase(
            id="scheduler_multi_step_list",
            description="List reminders after creation",
            prompt="What reminders do I have?",
            expected_behavior=(
                "The assistant should list the scheduled reminders, including "
                "the recurring standup at 10am EST on weekdays and the dentist "
                "appointment this Saturday at 2pm."
            ),
            criteria=[
                "Mentions the standup/daily reminder",
                "Mentions the dentist appointment",
                "Shows both reminders (not just one)",
            ],
        )

        response2 = await agent.process_message(
            list_case.prompt,
            session=session,
            user_id="eval-user",
        )

        await _judge_response(
            judge_llm, list_case, response2.text, response2.tool_calls
        )

        # --- Turn 3: Cancel the dentist reminder ---
        cancel_case = EvalCase(
            id="scheduler_multi_step_cancel",
            description="Cancel the dentist reminder",
            prompt="Cancel the dentist one",
            expected_behavior=(
                "The assistant should cancel the dentist appointment reminder "
                "while keeping the standup reminder active. It should confirm "
                "which reminder was cancelled."
            ),
            criteria=[
                "Cancels the dentist reminder specifically",
                "Confirms the cancellation",
                "Does not cancel the standup reminder",
            ],
        )

        response3 = await agent.process_message(
            cancel_case.prompt,
            session=session,
            user_id="eval-user",
        )

        await _judge_response(
            judge_llm, cancel_case, response3.text, response3.tool_calls
        )
