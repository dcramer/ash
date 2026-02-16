"""Person alias and relationship management."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ash.store.types import AliasEntry, PersonEntry, RelationshipClaim

if TYPE_CHECKING:
    from ash.store.store import Store


class PeopleRelationshipsMixin:
    """Alias and relationship operations for people."""

    async def add_alias(
        self: Store,
        person_id: str,
        alias: str,
        added_by: str | None = None,
    ) -> PersonEntry | None:
        person = self._graph.people.get(person_id)
        if not person:
            return None

        # Check if alias already exists
        if any(a.value.lower() == alias.lower() for a in person.aliases):
            return person

        now = datetime.now(UTC)
        person.aliases.append(
            AliasEntry(value=alias, added_by=added_by, created_at=now)
        )
        person.updated_at = now
        await self._persistence.save_people(self._graph.people)
        return person

    async def add_relationship(
        self: Store,
        person_id: str,
        relationship: str,
        stated_by: str | None = None,
        related_person_id: str | None = None,
    ) -> PersonEntry | None:
        person = self._graph.people.get(person_id)
        if not person:
            return None

        # Check if relationship already exists
        if any(
            r.relationship.lower() == relationship.lower() for r in person.relationships
        ):
            return person

        now = datetime.now(UTC)
        person.relationships.append(
            RelationshipClaim(
                relationship=relationship, stated_by=stated_by, created_at=now
            )
        )
        person.updated_at = now
        await self._persistence.save_people(self._graph.people)

        # Create HAS_RELATIONSHIP edge if we know the related person
        if related_person_id:
            from ash.graph.edges import create_has_relationship_edge

            edge = create_has_relationship_edge(
                person_id,
                related_person_id,
                relationship_type=relationship,
                stated_by=stated_by,
            )
            self._graph.add_edge(edge)
            await self._persistence.save_edges(self._graph.edges)

        return person
