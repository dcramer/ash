"""Tests for ash.core.signals."""

import pytest

from ash.core.signals import is_no_reply


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

    def test_embedded_in_text(self):
        assert is_no_reply("Here's the answer... [NO_REPLY]") is False

    def test_prefixed_text(self):
        assert is_no_reply("[NO_REPLY] and more") is False

    def test_lowercase(self):
        assert is_no_reply("[no_reply]") is False

    def test_mixed_case(self):
        assert is_no_reply("[No_Reply]") is False

    def test_missing_brackets(self):
        assert is_no_reply("NO_REPLY") is False

    def test_missing_open_bracket(self):
        assert is_no_reply("NO_REPLY]") is False

    def test_missing_close_bracket(self):
        assert is_no_reply("[NO_REPLY") is False

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
            "[NO_REPLY]\nBut also this",
            "```\n[NO_REPLY]\n```",
        ],
    )
    def test_not_standalone(self, text: str):
        assert is_no_reply(text) is False
