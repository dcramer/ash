"""Person management for the memory subsystem.

Handles person entity CRUD operations and resolution.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ash.memory.types import PersonEntry, PersonResolutionResult

if TYPE_CHECKING:
    from ash.memory.file_store import FileMemoryStore

RELATIONSHIP_TERMS = {
    "wife",
    "husband",
    "partner",
    "spouse",
    "mom",
    "mother",
    "dad",
    "father",
    "parent",
    "son",
    "daughter",
    "child",
    "kid",
    "brother",
    "sister",
    "sibling",
    "boss",
    "manager",
    "coworker",
    "colleague",
    "friend",
    "best friend",
    "roommate",
    "doctor",
    "therapist",
    "dentist",
}


class PersonManager:
    """Manages person entities for the memory system.

    Handles person lookup, creation, and resolution from references.
    """

    def __init__(self, store: FileMemoryStore) -> None:
        """Initialize person manager.

        Args:
            store: Filesystem-based memory store.
        """
        self._store = store

    async def find_person(
        self,
        owner_user_id: str,
        reference: str,
    ) -> PersonEntry | None:
        """Find a person by reference.

        Args:
            owner_user_id: User who owns the person record.
            reference: Name, relationship, or alias.

        Returns:
            Person entry or None.
        """
        return await self._store.find_person_by_reference(owner_user_id, reference)

    async def get_person(
        self,
        person_id: str,
        owner_user_id: str | None = None,
    ) -> PersonEntry | None:
        """Get person by ID.

        Args:
            person_id: Person UUID.
            owner_user_id: Optional owner filter.

        Returns:
            Person entry or None.
        """
        return await self._store.get_person(person_id, owner_user_id)

    async def get_known_people(self, owner_user_id: str) -> list[PersonEntry]:
        """Get all known people for a user.

        Args:
            owner_user_id: User to get people for.

        Returns:
            List of person entries.
        """
        return await self._store.get_people_for_user(owner_user_id)

    async def resolve_or_create_person(
        self,
        owner_user_id: str,
        reference: str,
        content_hint: str | None = None,
    ) -> PersonResolutionResult:
        """Resolve a reference to a person, creating if needed.

        Args:
            owner_user_id: User who will own the person record.
            reference: Name or relationship reference.
            content_hint: Content that may contain the person's name.

        Returns:
            Resolution result with person ID.
        """
        existing = await self._store.find_person_by_reference(owner_user_id, reference)
        if existing:
            return PersonResolutionResult(
                person_id=existing.id,
                created=False,
                person_name=existing.name,
            )

        name, relationship = self._parse_person_reference(reference, content_hint)

        person = await self._store.create_person(
            owner_user_id=owner_user_id,
            name=name,
            relationship=relationship,
            aliases=[reference] if reference.lower() != name.lower() else None,
        )

        return PersonResolutionResult(
            person_id=person.id,
            created=True,
            person_name=person.name,
        )

    async def resolve_person_names(self, person_ids: list[str]) -> dict[str, str]:
        """Resolve person IDs to names.

        Args:
            person_ids: List of person UUIDs.

        Returns:
            Dict mapping person_id to name.
        """
        result: dict[str, str] = {}
        for pid in person_ids:
            person = await self._store.get_person(pid)
            if person:
                result[pid] = person.name
        return result

    def _parse_person_reference(
        self,
        reference: str,
        content_hint: str | None = None,
    ) -> tuple[str, str | None]:
        """Parse a person reference into name and relationship."""
        ref_lower = reference.lower().strip()

        if ref_lower.startswith("@"):
            ref_lower = ref_lower[1:]

        relationship = ref_lower[3:] if ref_lower.startswith("my ") else None

        if relationship and relationship in RELATIONSHIP_TERMS:
            if content_hint:
                name = self._extract_name_from_content(content_hint, relationship)
                if name:
                    return name, relationship
            return relationship.title(), relationship

        return ref_lower.title(), relationship

    def _extract_name_from_content(
        self,
        content: str,
        relationship: str,
    ) -> str | None:
        """Try to extract a person's name from content."""
        match = re.search(
            rf"{relationship}(?:'s name is| is named) (\w+)", content, re.IGNORECASE
        )
        if match:
            return match.group(1)

        match = re.search(rf"(?:^|,\s*)my {relationship} (\w+)", content, re.IGNORECASE)
        if match:
            return match.group(1)

        match = re.search(r"^(\w+)'s\s", content)
        if match:
            name = match.group(1)
            if name.lower() not in ["user", "my", "the", "their", "his", "her"]:
                return name

        return None
