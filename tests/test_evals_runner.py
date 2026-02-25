from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

import evals.runner as runner
from ash.chats.history import read_recent_chat_history
from ash.core.session import SessionState
from evals.runner import (
    _record_eval_assistant_message,
    _record_eval_user_message,
    build_session_state,
    extract_tool_calls_from_session,
    load_eval_suite,
    run_yaml_eval_case,
)
from evals.types import (
    EvalCase,
    EvalConfig,
    EvalSuite,
    EvalTurn,
    JudgeResult,
    SessionConfig,
    SuiteDefaults,
    ToolInputAssertion,
)


def test_eval_runner_records_chat_history_entries() -> None:
    session = SessionState(
        session_id="eval-test",
        provider="eval",
        chat_id="chat-1",
        user_id="user-1",
    )
    session.context.username = "alice"
    session.context.display_name = "Alice"

    _record_eval_user_message(
        session,
        user_message="first message",
        user_id="user-1",
    )
    assert session.context.current_message_id is not None
    _record_eval_assistant_message(
        session,
        assistant_message="first response",
    )

    entries = read_recent_chat_history(provider="eval", chat_id="chat-1", limit=10)
    assert len(entries) == 2
    assert entries[0].role == "user"
    assert entries[0].content == "first message"
    assert entries[0].username == "alice"
    assert entries[1].role == "assistant"
    assert entries[1].content == "first response"


def test_load_eval_suite_infers_name_and_schema_version(tmp_path: Path) -> None:
    suite_path = tmp_path / "smoke_suite.yaml"
    suite_path.write_text("cases:\n  - id: c1\n    prompt: hello\n", encoding="utf-8")

    suite = load_eval_suite(suite_path)

    assert suite.name == "Smoke Suite"
    assert suite.schema_version == "2.0"
    assert suite.cases[0].id == "c1"


def test_load_eval_suite_rejects_unknown_keys(tmp_path: Path) -> None:
    suite_path = tmp_path / "bad_suite.yaml"
    suite_path.write_text(
        "name: Bad\ncases:\n  - id: c1\n    promt: typo\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_eval_suite(suite_path)


@pytest.mark.asyncio
async def test_run_yaml_eval_case_preserves_turn_tool_input_assertions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_turn_cases: list[EvalCase] = []

    async def _fake_execute_and_judge(*args, **kwargs):
        case = args[1]
        captured_turn_cases.append(case)
        return runner.EvalResult(
            case=case,
            response_text="ok",
            tool_calls=[],
            judge_result=JudgeResult(
                passed=True,
                score=1.0,
                reasoning="ok",
                criteria_scores={},
            ),
        )

    monkeypatch.setattr(runner, "_execute_and_judge", _fake_execute_and_judge)

    suite = EvalSuite(
        name="Test",
        defaults=SuiteDefaults(agent="default"),
        cases=[],
    )
    case = EvalCase(
        id="multi",
        turns=[EvalTurn(prompt="one"), EvalTurn(prompt="two")],
        tool_input_assertions=[
            ToolInputAssertion(tool="bash", input_contains=["--cron"])
        ],
    )
    components = SimpleNamespace(
        agent=object(),
        agent_executor=None,
        memory_manager=None,
    )

    await run_yaml_eval_case(
        components=cast(Any, components),
        suite=suite,
        case=case,
        judge_llm=cast(Any, SimpleNamespace()),
        config=EvalConfig(),
    )

    assert len(captured_turn_cases) == 2
    assert captured_turn_cases[0].tool_input_assertions
    assert captured_turn_cases[1].tool_input_assertions


def test_build_session_state_defaults_eval_provider_to_private_chat_type() -> None:
    defaults = SuiteDefaults(
        session=SessionConfig(provider="eval", user_id="u1"),
    )

    session = build_session_state(SessionConfig(), defaults, case_id="c1")

    assert session.context.chat_type == "private"


def test_extract_tool_calls_from_session_includes_tool_results() -> None:
    session_json = """
{
  "messages": [
    {
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_1",
          "name": "use_skill",
          "input": {"skill": "daily-brief", "message": "run"}
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "toolu_1",
          "content": "Daily brief ready",
          "is_error": false
        }
      ]
    }
  ]
}
"""
    calls = extract_tool_calls_from_session(session_json)
    assert len(calls) == 1
    assert calls[0]["name"] == "use_skill"
    assert calls[0]["id"] == "toolu_1"
    assert calls[0]["result"] == "Daily brief ready"
    assert calls[0]["is_error"] is False
