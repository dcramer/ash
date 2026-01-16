"""Eval execution utilities."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ash.core.agent import Agent
from ash.core.session import SessionState
from ash.llm.base import LLMProvider
from evals.judge import judge_response
from evals.types import EvalCase, EvalSuite, JudgeResult

logger = logging.getLogger(__name__)


def load_eval_suite(path: Path) -> EvalSuite:
    """Load an eval suite from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed EvalSuite.
    """
    with path.open() as f:
        data = yaml.safe_load(f)

    return EvalSuite.model_validate(data)


@dataclass
class EvalResult:
    """Result of running a single eval case."""

    case: EvalCase
    response_text: str
    tool_calls: list[dict[str, Any]]
    judge_result: JudgeResult
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Whether the eval passed."""
        return self.judge_result.passed and self.error is None

    @property
    def score(self) -> float:
        """Score from the judge."""
        return self.judge_result.score if self.error is None else 0.0


@dataclass
class EvalReport:
    """Report from running an eval suite."""

    suite_name: str
    results: list[EvalResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of cases."""
        return len(self.results)

    @property
    def passed(self) -> int:
        """Number of passed cases."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        """Number of failed cases."""
        return self.total - self.passed

    @property
    def accuracy(self) -> float:
        """Accuracy as a fraction (0.0 to 1.0)."""
        if not self.results:
            return 0.0
        return self.passed / self.total

    @property
    def average_score(self) -> float:
        """Average score across all cases."""
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / self.total

    def failed_cases(self) -> list[EvalResult]:
        """Get list of failed cases."""
        return [r for r in self.results if not r.passed]


async def run_eval_case(
    agent: Agent,
    case: EvalCase,
    judge_llm: LLMProvider,
    *,
    session: SessionState | None = None,
    judge_model: str = "claude-sonnet-4-5",
) -> EvalResult:
    """Run a single eval case and judge the result.

    Args:
        agent: The agent to test.
        case: The eval case to run.
        judge_llm: LLM provider to use for judging.
        session: Optional session state (creates fresh one if not provided).
        judge_model: Model to use for judging.

    Returns:
        EvalResult with the response and judgment.
    """
    # Create fresh session if not provided
    if session is None:
        session = SessionState(
            session_id=f"eval-{case.id}",
            provider="eval",
            chat_id="eval-chat",
            user_id="eval-user",
        )

    try:
        # Run the agent
        response = await agent.process_message(
            user_message=case.prompt,
            session=session,
            user_id="eval-user",
        )

        # Judge the response
        judge_result = await judge_response(
            llm=judge_llm,
            case=case,
            response_text=response.text,
            tool_calls=response.tool_calls,
            model=judge_model,
        )

        return EvalResult(
            case=case,
            response_text=response.text,
            tool_calls=response.tool_calls,
            judge_result=judge_result,
        )

    except Exception as e:
        logger.error(f"Eval case {case.id} failed with error: {e}")
        return EvalResult(
            case=case,
            response_text="",
            tool_calls=[],
            judge_result=JudgeResult(
                passed=False,
                score=0.0,
                reasoning=f"Execution error: {e}",
                criteria_scores={},
            ),
            error=str(e),
        )


async def run_eval_suite(
    agent: Agent,
    suite: EvalSuite,
    judge_llm: LLMProvider,
    *,
    judge_model: str = "claude-sonnet-4-5",
) -> EvalReport:
    """Run all cases in an eval suite.

    Args:
        agent: The agent to test.
        suite: The eval suite to run.
        judge_llm: LLM provider to use for judging.
        judge_model: Model to use for judging.

    Returns:
        EvalReport with all results.
    """
    report = EvalReport(suite_name=suite.name)

    for case in suite.cases:
        logger.info(f"Running eval case: {case.id}")
        result = await run_eval_case(
            agent=agent,
            case=case,
            judge_llm=judge_llm,
            judge_model=judge_model,
        )
        report.results.append(result)
        logger.info(
            f"Case {case.id}: {'PASSED' if result.passed else 'FAILED'} "
            f"(score: {result.score:.2f})"
        )

    return report
