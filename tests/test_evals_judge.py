from evals.judge import check_disallowed_tool_result_substrings
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
