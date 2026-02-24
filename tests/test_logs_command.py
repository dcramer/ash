import json

from ash.cli.commands.logs import _format_extras, query_logs


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
            "foo": "",
            "bar": None,
            "ok": "value",
        }
    )
    assert "context=" not in extras
    assert "session_id=" not in extras
    assert "foo=" not in extras
    assert "bar=" not in extras
    assert "ok=value" in extras


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
