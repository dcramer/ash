from ash.cli.commands.logs import _format_extras


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
