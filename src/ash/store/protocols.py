"""Protocol definitions for the store subsystem.

Defines abstract interfaces that can be implemented by different backends
or mocked in tests. These protocols enable dependency injection and testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ash.store.types import (
        MemoryEntry,
        MemoryType,
        PersonEntry,
        PersonResolutionResult,
        SearchResult,
        Sensitivity,
    )


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol for memory storage operations."""

    async def add_memory(
        self,
        content: str,
        source: str = "user",
        memory_type: MemoryType | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        subject_person_ids: list[str] | None = None,
        sensitivity: Sensitivity | None = None,
        portable: bool = True,
    ) -> MemoryEntry:
        """Add a memory entry."""
        ...

    async def get_memory(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory by ID."""
        ...

    async def delete_memory(
        self,
        memory_id: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> bool:
        """Delete a memory by ID."""
        ...

    async def list_memories(
        self,
        limit: int | None = 20,
        include_expired: bool = False,
        include_superseded: bool = False,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[MemoryEntry]:
        """List memories with filters."""
        ...


@runtime_checkable
class PersonStore(Protocol):
    """Protocol for person storage operations."""

    async def create_person(
        self,
        created_by: str,
        name: str,
        relationship: str | None = None,
        aliases: list[str] | None = None,
    ) -> PersonEntry:
        """Create a person entry."""
        ...

    async def get_person(self, person_id: str) -> PersonEntry | None:
        """Get a person by ID."""
        ...

    async def find_person(self, reference: str) -> PersonEntry | None:
        """Find a person by name, alias, or relationship."""
        ...

    async def list_people(self) -> list[PersonEntry]:
        """List all active people."""
        ...

    async def resolve_or_create_person(
        self,
        created_by: str,
        reference: str,
        content_hint: str | None = None,
        relationship_stated_by: str | None = None,
    ) -> PersonResolutionResult:
        """Resolve a reference to a person, creating if not found."""
        ...


@runtime_checkable
class SearchService(Protocol):
    """Protocol for semantic search operations."""

    async def search(
        self,
        query: str,
        limit: int = 10,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[SearchResult]:
        """Search memories by semantic similarity."""
        ...


@runtime_checkable
class LLMService(Protocol):
    """Protocol for LLM-powered operations in the store.

    Used for fuzzy matching and supersession verification.
    """

    async def fuzzy_match_person(
        self,
        reference: str,
        candidates: list[PersonEntry],
        speaker: str | None = None,
    ) -> PersonEntry | None:
        """Find a person using LLM-based fuzzy matching."""
        ...

    async def verify_supersession(
        self,
        old_memory: MemoryEntry,
        new_memory: MemoryEntry,
    ) -> bool:
        """Verify if new_memory supersedes old_memory."""
        ...
