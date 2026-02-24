"""Tests for Telegram checkpoint UI text formatting."""

from ash.providers.telegram.checkpoint_ui import format_checkpoint_message


def test_format_checkpoint_message_without_options() -> None:
    result = format_checkpoint_message({"prompt": "Proceed?"})
    assert result == "Proceed?"


def test_format_checkpoint_message_includes_suggested_responses() -> None:
    result = format_checkpoint_message(
        {"prompt": "Choose next step", "options": ["Proceed", "Cancel"]}
    )
    assert result == "Choose next step\n\nSuggested responses: Proceed, Cancel"
