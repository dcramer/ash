import json

from ash.cli.commands.logs import _format_context_marker, _format_extras, query_logs


def test_format_extras_keeps_more_error_message_detail() -> None:
    extras = _format_extras(
        {
            "message": "event",
            "component": "telegram",
            "error.message": "e" * 280,
        }
    )
    assert "error.message=" in extras
    # Should not use the generic short truncation budget.
    assert "..." not in extras


def test_format_extras_truncates_generic_fields() -> None:
    extras = _format_extras(
        {
            "message": "event",
            "component": "telegram",
            "misc": "m" * 200,
        }
    )
    assert "misc=" in extras
    assert "..." in extras


def test_format_extras_keeps_more_ids_detail() -> None:
    extras = _format_extras(
        {
            "message": "event",
            "component": "store",
            "memory.ids": "i" * 250,
        }
    )
    # ids fields get a larger budget than generic fields.
    assert "memory.ids=" in extras
    assert "..." not in extras


def test_format_extras_hides_empty_and_context_noise_fields() -> None:
    extras = _format_extras(
        {
            "message": "event",
            "component": "telegram",
            "context": "",
            "context ": "",
            "session_id": "s-1",
            "agent_name": "main",
            "foo": "",
            "bar": None,
            "ok": "value",
        }
    )
    assert "context=" not in extras
    assert "session_id=" not in extras
    assert "agent_name=" not in extras
    assert "foo=" not in extras
    assert "bar=" not in extras
    assert "ok=value" in extras


def test_format_context_marker_compact_display() -> None:
    marker = _format_context_marker(
        {
            "chat_id": "-542863895",
            "session_id": "telegram_-542863895_1662",
            "thread_id": "1662",
            "agent_name": "main",
            "provider": "telegram",
            "user_id": "1234567890",
            "chat_type": "group",
            "source_username": "dcramer",
        }
    )

    assert marker.startswith("[")
    assert "-5428638" in marker
    assert "s:telegram" in marker
    assert "t:1662" in marker
    assert "@main" in marker
    assert "p:telegram" in marker
    assert "u:12345678" in marker
    assert "ct:group" in marker
    assert "src:dcramer" in marker


def test_format_extras_summarizes_id_lists() -> None:
    extras = _format_extras(
        {
            "message": "event",
            "component": "core",
            "memory.ids": ["abc123456", "def234567", "ghi345678", "jkl456789"],
        }
    )
    assert "memory.ids=4 ids[abc12345, def23456, ghi34567, ...]" in extras


def test_format_extras_summarizes_tool_arguments_without_newlines() -> None:
    extras = _format_extras(
        {
            "message": "tool_executed",
            "component": "tools",
            "gen_ai.tool.call.arguments": {
                "command": "ash-sb memory search 'watch'\n--this-chat",
                "timeout": 60,
                "foo": "bar",
            },
        }
    )
    assert "gen_ai.tool.call.arguments={" in extras
    assert "command=ash-sb memory search 'watch' --this-chat" in extras
    assert "+1 more" in extras
    assert "\n" not in extras


def test_query_logs_returns_latest_entries_with_newest_last(tmp_path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    log_file = logs_dir / "2026-02-24.jsonl"
    entries = [
        {
            "ts": "2026-02-24T02:37:21Z",
            "level": "INFO",
            "component": "core",
            "message": "first",
        },
        {
            "ts": "2026-02-24T02:37:22Z",
            "level": "INFO",
            "component": "core",
            "message": "second",
        },
        {
            "ts": "2026-02-24T02:37:23Z",
            "level": "INFO",
            "component": "core",
            "message": "third",
        },
    ]
    log_file.write_text("".join(json.dumps(entry) + "\n" for entry in entries))

    results = query_logs(logs_dir, limit=2)

    assert [entry["message"] for entry in results] == ["second", "third"]
