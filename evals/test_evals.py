"""YAML-driven evaluation tests.

Single parametrized test file that discovers all v2.0 YAML eval suites
and runs each case through the generic orchestration engine.

Run with: uv run pytest evals/test_evals.py -v -s -m eval
"""

from pathlib import Path

import pytest

from ash.core.types import AgentComponents
from ash.llm.base import LLMProvider
from evals.runner import load_eval_suite, run_yaml_eval_case

CASES_DIR = Path(__file__).parent / "cases"


def _discover_cases(
    agent_type: str,
) -> list[tuple[str, str, Path]]:
    """Discover (suite_stem, case_id, suite_path) tuples for a given agent type."""
    cases: list[tuple[str, str, Path]] = []
    if not CASES_DIR.exists():
        return cases

    for path in sorted(CASES_DIR.glob("*.yaml")):
        suite = load_eval_suite(path)
        for case in suite.cases:
            effective_agent = case.agent or suite.defaults.agent
            if effective_agent == agent_type:
                cases.append((path.stem, case.id, path))

    return cases


_DEFAULT_CASES = _discover_cases("default")
_MEMORY_CASES = _discover_cases("memory")


@pytest.mark.eval
class TestYamlEvals:
    """YAML-driven evaluation tests."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("suite_stem", "case_id", "suite_path"),
        _DEFAULT_CASES,
        ids=[f"{stem}::{cid}" for stem, cid, _ in _DEFAULT_CASES],
    )
    async def test_eval_case(
        self,
        suite_stem: str,
        case_id: str,
        suite_path: Path,
        eval_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        suite = load_eval_suite(suite_path)
        case = next(c for c in suite.cases if c.id == case_id)

        results = await run_yaml_eval_case(
            components=eval_agent,
            suite=suite,
            case=case,
            judge_llm=judge_llm,
        )

        for result in results:
            assert result.passed, (
                f"{result.case.id} failed (score={result.score}): "
                f"{result.judge_result.reasoning}"
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("suite_stem", "case_id", "suite_path"),
        _MEMORY_CASES,
        ids=[f"{stem}::{cid}" for stem, cid, _ in _MEMORY_CASES],
    )
    async def test_memory_eval_case(
        self,
        suite_stem: str,
        case_id: str,
        suite_path: Path,
        eval_memory_agent: AgentComponents,
        judge_llm: LLMProvider,
    ) -> None:
        suite = load_eval_suite(suite_path)
        case = next(c for c in suite.cases if c.id == case_id)

        results = await run_yaml_eval_case(
            components=eval_memory_agent,
            suite=suite,
            case=case,
            judge_llm=judge_llm,
        )

        for result in results:
            assert result.passed, (
                f"{result.case.id} failed (score={result.score}): "
                f"{result.judge_result.reasoning}"
            )
