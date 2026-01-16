"""Eval execution utilities."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ash.core.agent import Agent
from ash.core.session import SessionState
from ash.llm.base import LLMProvider
from evals.judge import LLMJudge
from evals.types import EvalCase, EvalConfig, EvalSuite, JudgeResult

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


def discover_eval_suites(cases_dir: Path | None = None) -> list[Path]:
    """Auto-discover eval suite YAML files.

    Args:
        cases_dir: Directory to search (defaults to evals/cases).

    Returns:
        List of paths to YAML suite files.
    """
    if cases_dir is None:
        cases_dir = Path(__file__).parent / "cases"

    if not cases_dir.exists():
        logger.warning(f"Cases directory not found: {cases_dir}")
        return []

    suites = list(cases_dir.glob("*.yaml")) + list(cases_dir.glob("*.yml"))
    return sorted(suites)


def get_case_by_id(suite: EvalSuite, case_id: str) -> EvalCase:
    """Get a specific case from a suite by ID.

    Args:
        suite: The eval suite.
        case_id: ID of the case to find.

    Returns:
        The matching EvalCase.

    Raises:
        ValueError: If no case with the given ID exists.
    """
    for case in suite.cases:
        if case.id == case_id:
            return case
    raise ValueError(f"Case '{case_id}' not found in suite '{suite.name}'")


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
        """Whether the eval passed (excludes judge errors)."""
        return self.judge_result.passed and self.error is None

    @property
    def score(self) -> float:
        """Score from the judge."""
        return self.judge_result.score if self.error is None else 0.0

    @property
    def is_judge_error(self) -> bool:
        """Whether this result is due to a judge error, not an actual failure."""
        return self.judge_result.judge_error


@dataclass
class EvalReport:
    """Report from running an eval suite."""

    suite_name: str
    config: EvalConfig = field(default_factory=EvalConfig)
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
        """Number of failed cases (excluding judge errors)."""
        return sum(1 for r in self.results if not r.passed and not r.is_judge_error)

    @property
    def judge_errors(self) -> int:
        """Number of cases that failed due to judge errors."""
        return sum(1 for r in self.results if r.is_judge_error)

    @property
    def accuracy(self) -> float:
        """Accuracy as a fraction (0.0 to 1.0).

        Excludes judge errors from the calculation to give true accuracy.
        """
        valid_results = [r for r in self.results if not r.is_judge_error]
        if not valid_results:
            return 0.0
        passed = sum(1 for r in valid_results if r.passed)
        return passed / len(valid_results)

    @property
    def average_score(self) -> float:
        """Average score across all cases (excluding judge errors)."""
        valid_results = [r for r in self.results if not r.is_judge_error]
        if not valid_results:
            return 0.0
        return sum(r.score for r in valid_results) / len(valid_results)

    def failed_cases(self) -> list[EvalResult]:
        """Get list of failed cases (excluding judge errors)."""
        return [r for r in self.results if not r.passed and not r.is_judge_error]

    def judge_error_cases(self) -> list[EvalResult]:
        """Get list of cases that failed due to judge errors."""
        return [r for r in self.results if r.is_judge_error]


async def run_eval_case(
    agent: Agent,
    case: EvalCase,
    judge_llm: LLMProvider,
    *,
    session: SessionState | None = None,
    config: EvalConfig | None = None,
    judge_model: str | None = None,
) -> EvalResult:
    """Run a single eval case and judge the result.

    Args:
        agent: The agent to test.
        case: The eval case to run.
        judge_llm: LLM provider to use for judging.
        session: Optional session state (creates fresh one if not provided).
        config: Eval configuration (uses defaults if not provided).
        judge_model: Override judge model (deprecated, use config instead).

    Returns:
        EvalResult with the response and judgment.
    """
    if config is None:
        config = EvalConfig()
    if judge_model is not None:
        # Support legacy judge_model parameter
        config = EvalConfig(
            judge_model=judge_model,
            judge_temperature=config.judge_temperature,
            judge_max_tokens=config.judge_max_tokens,
            retry_attempts=config.retry_attempts,
            retry_base_delay=config.retry_base_delay,
        )

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
        judge = LLMJudge(judge_llm, config)
        judge_result = await judge.evaluate(
            case=case,
            response_text=response.text,
            tool_calls=response.tool_calls,
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
    config: EvalConfig | None = None,
    judge_model: str | None = None,
) -> EvalReport:
    """Run all cases in an eval suite.

    Args:
        agent: The agent to test.
        suite: The eval suite to run.
        judge_llm: LLM provider to use for judging.
        config: Eval configuration.
        judge_model: Override judge model (deprecated, use config instead).

    Returns:
        EvalReport with all results.
    """
    if config is None:
        config = EvalConfig()
    if judge_model is not None:
        config = EvalConfig(
            judge_model=judge_model,
            judge_temperature=config.judge_temperature,
            judge_max_tokens=config.judge_max_tokens,
            retry_attempts=config.retry_attempts,
            retry_base_delay=config.retry_base_delay,
        )

    report = EvalReport(suite_name=suite.name, config=config)

    for case in suite.cases:
        logger.info(f"Running eval case: {case.id}")
        result = await run_eval_case(
            agent=agent,
            case=case,
            judge_llm=judge_llm,
            config=config,
        )
        report.results.append(result)

        status = (
            "PASSED"
            if result.passed
            else ("JUDGE_ERROR" if result.is_judge_error else "FAILED")
        )
        logger.info(f"Case {case.id}: {status} (score: {result.score:.2f})")

    # Log summary
    if report.judge_errors > 0:
        logger.warning(
            f"Suite '{suite.name}': {report.judge_errors} cases had judge errors"
        )

    return report
