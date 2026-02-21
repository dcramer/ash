from ash.chats.history import read_recent_chat_history
from ash.core.session import SessionState
from evals.runner import _record_eval_assistant_message, _record_eval_user_message


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
