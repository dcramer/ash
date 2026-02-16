"""Person CRUD operations: create, read, update, delete."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ash.store.types import (
    AliasEntry,
    PersonEntry,
    RelationshipClaim,
)

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)


class PeopleCrudMixin:
    """Person create, read, update, delete operations."""

    async def create_person(
        self: Store,
        created_by: str,
        name: str,
        relationship: str | None = None,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        relationship_stated_by: str | None = None,
    ) -> PersonEntry:
        now = datetime.now(UTC)
        person_id = str(uuid.uuid4())

        relationships: list[RelationshipClaim] = []
        if relationship:
            relationships.append(
                RelationshipClaim(
                    relationship=relationship,
                    stated_by=relationship_stated_by or created_by,
                    created_at=now,
                )
            )
        alias_entries = [
            AliasEntry(value=a, added_by=created_by, created_at=now)
            for a in (aliases or [])
        ]

        entry = PersonEntry(
            id=person_id,
            version=1,
            created_by=created_by,
            name=name,
            relationships=relationships,
            aliases=alias_entries,
            created_at=now,
            updated_at=now,
            metadata=metadata,
        )

        self._graph.add_person(entry)
        await self._persistence.save_people(self._graph.people)

        logger.debug(
            "person_created", extra={"person_id": entry.id, "person_name": name}
        )
        return entry

    async def get_person(self: Store, person_id: str) -> PersonEntry | None:
        return self._graph.people.get(person_id)

    async def list_people(self: Store, limit: int | None = None) -> list[PersonEntry]:
        from ash.graph.edges import get_merged_into

        people = [
            p
            for p in self._graph.people.values()
            if get_merged_into(self._graph, p.id) is None
        ]
        people.sort(
            key=lambda p: p.updated_at or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )
        if limit is not None:
            people = people[:limit]
        return people

    async def get_all_people(self: Store) -> list[PersonEntry]:
        people = list(self._graph.people.values())
        people.sort(
            key=lambda p: p.updated_at or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )
        return people

    async def update_person(
        self: Store,
        person_id: str,
        name: str | None = None,
        updated_by: str | None = None,
        clear_merged: bool = False,
    ) -> PersonEntry | None:
        person = self._graph.people.get(person_id)
        if not person:
            return None

        now = datetime.now(UTC)
        person.updated_at = now
        if name is not None:
            person.name = name
        if clear_merged:
            from ash.graph.edges import MERGED_INTO

            # Remove MERGED_INTO edges pointing from this person
            for edge in self._graph.get_outgoing(person_id, edge_type=MERGED_INTO):
                self._graph.remove_edge(edge.id)
            await self._persistence.save_edges(self._graph.edges)

        await self._persistence.save_people(self._graph.people)
        return person

    async def get_person_names_batch(
        self: Store, person_ids: list[str]
    ) -> dict[str, str]:
        if not person_ids:
            return {}
        result: dict[str, str] = {}
        for pid in person_ids:
            person = self._graph.people.get(pid)
            if person:
                result[pid] = person.name
        return result

    async def delete_person(self: Store, person_id: str) -> bool:
        person = self._graph.people.get(person_id)
        if not person:
            return False

        name = person.name

        # Clear MERGED_INTO edges pointing to this person
        from ash.graph.edges import MERGED_INTO

        for edge in self._graph.get_incoming(person_id, edge_type=MERGED_INTO):
            self._graph.remove_edge(edge.id)
        await self._persistence.save_edges(self._graph.edges)

        self._graph.remove_person(person_id)
        await self._persistence.save_people(self._graph.people)

        logger.debug(
            "person_deleted",
            extra={"person_id": person_id, "person_name": name},
        )
        return True
