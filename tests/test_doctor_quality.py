"""Tests for memory doctor quality review helpers."""

from ash.cli.commands.memory.doctor._helpers import (
    is_trivial_rewrite as _is_trivial_rewrite,
)
from ash.cli.commands.memory.doctor.quality import (
    _is_unstable_subject_swap as _is_unstable_subject_swap,
)


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


class TestUnstableSubjectSwap:
    def test_detects_primary_subject_flip(self):
        assert (
            _is_unstable_subject_swap(
                "David Cramer owns a Rolex Daytona that was gifted by Sukhpreet.",
                "Sukhpreet Sembhi owns a Rolex Daytona that was gifted by David.",
                ["David Cramer", "Sukhpreet Sembhi"],
            )
            is True
        )

    def test_allows_user_to_named_subject_rewrite(self):
        assert (
            _is_unstable_subject_swap(
                "User lives in San Francisco.",
                "David Cramer lives in San Francisco.",
                ["David Cramer"],
            )
            is False
        )
