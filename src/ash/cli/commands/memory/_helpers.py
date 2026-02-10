"""Shared helpers for memory CLI commands."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ash.memory.file_store import FileMemoryStore


def get_memory_store() -> "FileMemoryStore":
    """Get a configured FileMemoryStore instance.

    This factory function allows for easier testing and future
    path injection if needed.
    """
    from ash.memory.file_store import FileMemoryStore

    return FileMemoryStore()


def matches_person(source_id: str, person) -> bool:
    """Check if source_id matches a person's name or aliases."""
    if person.name.lower() == source_id:
        return True
    for alias in person.aliases or []:
        if alias.lower() == source_id:
            return True
    return False


def is_source_self_reference(
    source_user_id: str | None,
    owner_user_id: str | None,
    subject_person_ids: list[str] | None,
    all_people: list,
    people_by_id: dict,
) -> bool:
    """Determine if the source user is speaking about themselves.

    Returns True if:
    - source matches a self-person (owner speaking about themselves)
    - source matches any subject person (source is the subject)
    """
    if not source_user_id:
        return False

    source_id = source_user_id.lower()

    # Check if source matches a self-person
    for person in all_people:
        if person.owner_user_id == owner_user_id and person.relationship == "self":
            if matches_person(source_id, person):
                return True

    # Check if source matches any subject person
    if subject_person_ids:
        for person_id in subject_person_ids:
            person = people_by_id.get(person_id)
            if person and matches_person(source_id, person):
                return True

    return False
