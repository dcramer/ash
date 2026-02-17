"""Evaluation test suite for testing chat behaviors with LLM-as-judge."""

from evals.judge import Judge, LLMJudge, judge_response
from evals.report import print_report
from evals.runner import (
    EvalReport,
    EvalResult,
    build_session_state,
    check_structural_assertions,
    discover_eval_suites,
    drain_extraction_tasks,
    get_case_by_id,
    load_eval_suite,
    run_eval_case,
    run_eval_suite,
    run_setup_steps,
    run_yaml_eval_case,
)
from evals.types import (
    Assertions,
    EvalCase,
    EvalConfig,
    EvalSuite,
    EvalTurn,
    JudgeResult,
    MemoryAssertion,
    PersonAssertion,
    SessionConfig,
    SetupStep,
    SuiteDefaults,
)

__all__ = [
    # Types
    "Assertions",
    "EvalCase",
    "EvalConfig",
    "EvalResult",
    "EvalReport",
    "EvalSuite",
    "EvalTurn",
    "JudgeResult",
    "MemoryAssertion",
    "PersonAssertion",
    "SessionConfig",
    "SetupStep",
    "SuiteDefaults",
    # Judge
    "Judge",
    "LLMJudge",
    "judge_response",
    # Runner
    "build_session_state",
    "check_structural_assertions",
    "discover_eval_suites",
    "drain_extraction_tasks",
    "get_case_by_id",
    "load_eval_suite",
    "run_eval_case",
    "run_eval_suite",
    "run_setup_steps",
    "run_yaml_eval_case",
    # Report
    "print_report",
]
