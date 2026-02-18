"""Tests for ash.core.signals."""

import pytest

from ash.core.signals import contains_no_reply, is_no_reply


class TestIsNoReply:
    def test_exact_match(self):
        assert is_no_reply("[NO_REPLY]") is True

    def test_with_leading_whitespace(self):
        assert is_no_reply("  [NO_REPLY]") is True

    def test_with_trailing_whitespace(self):
        assert is_no_reply("[NO_REPLY]  ") is True

    def test_with_surrounding_whitespace(self):
        assert is_no_reply("  [NO_REPLY]  ") is True

    def test_with_newlines(self):
        assert is_no_reply("\n[NO_REPLY]\n") is True

    def test_without_brackets(self):
        assert is_no_reply("NO_REPLY") is True

    def test_missing_open_bracket(self):
        assert is_no_reply("NO_REPLY]") is True

    def test_missing_close_bracket(self):
        assert is_no_reply("[NO_REPLY") is True

    def test_lowercase(self):
        assert is_no_reply("[no_reply]") is True

    def test_mixed_case(self):
        assert is_no_reply("[No_Reply]") is True

    def test_lowercase_no_brackets(self):
        assert is_no_reply("no_reply") is True

    def test_with_trailing_reasoning(self):
        """Leaked reasoning after the token still counts as NO_REPLY."""
        assert is_no_reply("[NO_REPLY]\n(I don't have anything to add here)") is True

    def test_with_trailing_reasoning_multiline(self):
        assert (
            is_no_reply("[NO_REPLY]\nThe conversation doesn't need my input.") is True
        )

    def test_blank_lines_before_token(self):
        assert is_no_reply("\n\n  \n[NO_REPLY]") is True

    def test_embedded_in_text(self):
        assert is_no_reply("Here's the answer... [NO_REPLY]") is False

    def test_prefixed_text(self):
        assert is_no_reply("Sure, [NO_REPLY]") is False

    def test_markdown_code_wrapped(self):
        assert is_no_reply("`[NO_REPLY]`") is False

    def test_markdown_bold_wrapped(self):
        assert is_no_reply("**[NO_REPLY]**") is False

    def test_empty_string(self):
        assert is_no_reply("") is False

    def test_whitespace_only(self):
        assert is_no_reply("   ") is False

    def test_token_constant(self):
        from ash.core.signals import NO_REPLY

        assert NO_REPLY == "[NO_REPLY]"

    @pytest.mark.parametrize(
        "text",
        [
            "Sure, [NO_REPLY]",
            "```\n[NO_REPLY]\n```",
        ],
    )
    def test_not_standalone(self, text: str):
        assert is_no_reply(text) is False

    def test_regression_parenthetical_leak(self):
        """Regression: LLM outputs NO_REPLY with trailing parenthetical."""
        assert (
            is_no_reply("[NO_REPLY] (this conversation doesn't need my input)") is False
        )
        assert (
            is_no_reply("[NO_REPLY]\n(this conversation doesn't need my input)") is True
        )


class TestContainsNoReply:
    def test_exact_match(self):
        assert contains_no_reply("[NO_REPLY]") is True

    def test_no_brackets(self):
        assert contains_no_reply("NO_REPLY") is True

    def test_lowercase(self):
        assert contains_no_reply("[no_reply]") is True

    def test_within_text(self):
        assert contains_no_reply("Some text [NO_REPLY] more text") is True

    def test_with_trailing_reasoning(self):
        assert contains_no_reply("[NO_REPLY]\n(I don't have anything to add)") is True

    def test_no_match(self):
        assert contains_no_reply("Hello, how are you?") is False

    def test_empty_string(self):
        assert contains_no_reply("") is False

    def test_partial_match(self):
        assert contains_no_reply("NO_REP") is False
