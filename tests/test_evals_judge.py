from evals.judge import (
    check_disallowed_tool_result_substrings,
    check_tool_input_assertions,
)
from evals.types import EvalCase


def test_disallowed_tool_result_substrings_passes_when_absent() -> None:
    case = EvalCase(id="c1", prompt="p")
    result = check_disallowed_tool_result_substrings(
        case,
        [{"name": "bash", "result": "ok", "is_error": False}],
    )
    assert result is None


def test_disallowed_tool_result_substrings_fails_on_match() -> None:
    case = EvalCase(
        id="c1",
        prompt="p",
        disallowed_tool_result_substrings=["Message not found in session"],
    )
    result = check_disallowed_tool_result_substrings(
        case,
        [
            {
                "name": "bash",
                "result": "Exit code 1: Extraction failed: Message not found in session",
                "is_error": True,
            }
        ],
    )
    assert result is not None
    assert result.passed is False
    assert "Disallowed tool result content" in result.reasoning


def test_tool_input_assertions_pass() -> None:
    case = EvalCase(
        id="c2",
        prompt="p",
        tool_input_assertions=[
            {
                "tool": "bash",
                "input_contains": ["ash-sb schedule create", "--cron"],
                "min_calls": 1,
            }
        ],
    )
    result = check_tool_input_assertions(
        case,
        [
            {
                "name": "bash",
                "input": {
                    "command": "ash-sb schedule create 'check score' --cron '*/1 * * * *'"
                },
            }
        ],
    )
    assert result is None


def test_tool_input_assertions_fail_on_missing_substring() -> None:
    case = EvalCase(
        id="c3",
        prompt="p",
        tool_input_assertions=[
            {
                "tool": "bash",
                "input_contains": ["--cron"],
                "min_calls": 1,
            }
        ],
    )
    result = check_tool_input_assertions(
        case,
        [
            {
                "name": "bash",
                "input": {
                    "command": "ash-sb schedule create 'check score' --at 'in 1 minute'"
                },
            }
        ],
    )
    assert result is not None
    assert result.passed is False
    assert "Tool input assertion failures" in result.reasoning
