"""Tests for memory doctor quality review helpers."""

from ash.cli.commands.memory.doctor import _is_trivial_rewrite


class TestIsTrivialRewrite:
    """Tests for the _is_trivial_rewrite filter."""

    def test_identical_content(self):
        """Identical strings are trivial."""
        assert _is_trivial_rewrite("User likes coffee", "User likes coffee") is True

    def test_case_only_change(self):
        """Capitalization-only changes are trivial."""
        assert _is_trivial_rewrite("User Likes Coffee", "user likes coffee") is True

    def test_filler_word_in_long_text(self):
        """Removing a filler word in longer text stays above threshold."""
        assert (
            _is_trivial_rewrite(
                "David really enjoys hiking in the mountains on weekends",
                "David enjoys hiking in the mountains on weekends",
            )
            is True
        )

    def test_whitespace_normalization(self):
        """Extra whitespace differences are trivial."""
        assert _is_trivial_rewrite("User  likes   coffee", "User likes coffee") is True

    def test_punctuation_differences(self):
        """Punctuation-only differences are trivial."""
        assert _is_trivial_rewrite("User likes coffee.", "User likes coffee") is True

    def test_substantive_rewrite_adding_subject(self):
        """Adding a subject name is a substantive rewrite."""
        assert (
            _is_trivial_rewrite(
                "Birthday is August 12", "David's birthday is August 12"
            )
            is False
        )

    def test_substantive_rewrite_perspective_fix(self):
        """Fixing perspective is a substantive rewrite."""
        assert (
            _is_trivial_rewrite(
                "Your favorite color is blue", "David's favorite color is blue"
            )
            is False
        )

    def test_substantive_rewrite_adding_context(self):
        """Adding meaningful context is a substantive rewrite."""
        assert (
            _is_trivial_rewrite("Likes hiking", "David enjoys hiking in the Cascades")
            is False
        )

    def test_completely_different_content(self):
        """Completely different content is not trivial."""
        assert _is_trivial_rewrite("Likes coffee", "Has a dog named Max") is False

    def test_censored_language_is_substantive(self):
        """Softening/censoring language is a substantive change, not trivial."""
        assert (
            _is_trivial_rewrite("Has shitty tests", "Has poor quality tests") is False
        )
