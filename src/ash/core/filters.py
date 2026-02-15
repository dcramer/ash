"""Filtering utilities for agent message processing.

Contains filters for owner name detection and other extraction-time filtering.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OwnerMatchers:
    """Compiled matchers for owner name filtering.

    Used to detect when a subject name refers to the speaker (owner)
    rather than a third party, to avoid creating duplicate person entries.
    """

    exact: set[str]  # Exact matches (username, full name)
    parts: set[str]  # Name parts (first name, last name)


def build_owner_matchers(owner_names: list[str] | None) -> OwnerMatchers:
    """Build matchers for filtering owner names from subjects.

    Matches exact full names and usernames. For multi-word display names,
    also matches individual name parts (e.g., "David" or "Cramer" from
    "David Cramer") to catch LLM extractions that use partial names.
    Single-word names and usernames are exact-only to avoid false positives.

    Prefers false positives (dropping a valid subject) over false negatives
    (creating a duplicate person entry for the speaker), since duplicate
    identity entries cause persistent downstream issues.

    Args:
        owner_names: List of names/handles that refer to the owner.

    Returns:
        OwnerMatchers with exact and part match sets.
    """
    exact: set[str] = set()
    parts: set[str] = set()

    if owner_names:
        for name in owner_names:
            cleaned = name.lower().lstrip("@")
            exact.add(cleaned)
            # For multi-word names, add each word as a part match
            words = cleaned.split()
            if len(words) > 1:
                for word in words:
                    if len(word) >= 3:  # Skip very short parts (initials, etc.)
                        parts.add(word)

    return OwnerMatchers(exact=exact, parts=parts)


def is_owner_name(subject: str, matchers: OwnerMatchers) -> bool:
    """Check if a subject name refers to the owner.

    Uses exact matching, part matching, and prefix checks to catch common
    LLM extraction variations (first name only, nicknames, abbreviations).
    Prefers false positives over creating duplicate person entries.

    Args:
        subject: The subject name to check.
        matchers: Pre-compiled owner matchers.

    Returns:
        True if the subject likely refers to the owner.
    """
    normalized = subject.lower().lstrip("@")

    if normalized in matchers.exact or normalized in matchers.parts:
        return True

    # Skip very short subjects to avoid false positives on initials
    if len(normalized) < 3:
        return False

    # Prefix checks: catch nicknames and abbreviations.
    # "Dave" matches part "david", "David C." starts with part "david".
    for part in matchers.parts:
        if part.startswith(normalized) or normalized.startswith(part):
            return True

    return False


class OwnerNameFilter:
    """Filter for detecting owner name references.

    Provides a high-level API for checking if a subject name refers to
    the owner (speaker) rather than a third party.
    """

    def __init__(self, owner_names: list[str] | None = None):
        """Initialize filter with owner identifiers.

        Args:
            owner_names: Names/handles that refer to the owner
                (e.g., username, display name).
        """
        self._matchers = build_owner_matchers(owner_names)

    def is_owner_reference(self, name: str) -> bool:
        """Check if a name refers to the owner.

        Args:
            name: The name to check.

        Returns:
            True if the name likely refers to the owner.
        """
        return is_owner_name(name, self._matchers)

    @property
    def owner_names(self) -> set[str]:
        """Get the set of exact owner name matches."""
        return self._matchers.exact
