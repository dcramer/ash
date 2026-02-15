"""Person CRUD operations: create, read, update, delete."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from ash.store.mappers import row_to_person as _row_to_person
from ash.store.people.helpers import load_person_full
from ash.store.types import (
    AliasEntry,
    PersonEntry,
    RelationshipClaim,
    _parse_datetime,
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

        async with self._db.session() as session:
            await session.execute(
                text("""
                    INSERT INTO people (id, version, created_by, name, created_at, updated_at, metadata)
                    VALUES (:id, 1, :created_by, :name, :created_at, :updated_at, :metadata)
                """),
                {
                    "id": person_id,
                    "created_by": created_by,
                    "name": name,
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                    "metadata": json.dumps(metadata) if metadata else None,
                },
            )
            for rc in relationships:
                await session.execute(
                    text("""
                        INSERT INTO person_relationships (person_id, relationship, stated_by, created_at)
                        VALUES (:pid, :rel, :stated_by, :created_at)
                    """),
                    {
                        "pid": person_id,
                        "rel": rc.relationship,
                        "stated_by": rc.stated_by,
                        "created_at": rc.created_at.isoformat()
                        if rc.created_at
                        else None,
                    },
                )
            for ae in alias_entries:
                await session.execute(
                    text("""
                        INSERT INTO person_aliases (person_id, value, added_by, created_at)
                        VALUES (:pid, :value, :added_by, :created_at)
                    """),
                    {
                        "pid": person_id,
                        "value": ae.value,
                        "added_by": ae.added_by,
                        "created_at": ae.created_at.isoformat()
                        if ae.created_at
                        else None,
                    },
                )

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
        logger.debug(
            "person_created", extra={"person_id": entry.id, "person_name": name}
        )
        return entry

    async def get_person(self: Store, person_id: str) -> PersonEntry | None:
        async with self._db.session() as session:
            return await load_person_full(session, person_id)

    async def list_people(self: Store, limit: int | None = None) -> list[PersonEntry]:
        async with self._db.session() as session:
            query = "SELECT * FROM people WHERE merged_into IS NULL ORDER BY updated_at DESC"
            params: dict[str, Any] = {}
            if limit is not None:
                query += " LIMIT :limit"
                params["limit"] = limit

            result = await session.execute(text(query), params)
            rows = result.fetchall()
            if not rows:
                return []

            person_ids = [row.id for row in rows]

            # Batch-load aliases
            aliases_map: dict[str, list[AliasEntry]] = {pid: [] for pid in person_ids}
            placeholders = ", ".join(f":id{i}" for i in range(len(person_ids)))
            id_params = {f"id{i}": pid for i, pid in enumerate(person_ids)}
            alias_result = await session.execute(
                text(
                    f"SELECT person_id, value, added_by, created_at FROM person_aliases WHERE person_id IN ({placeholders})"
                ),
                id_params,
            )
            for arow in alias_result.fetchall():
                aliases_map[arow[0]].append(
                    AliasEntry(
                        value=arow[1],
                        added_by=arow[2],
                        created_at=_parse_datetime(arow[3]),
                    )
                )

            # Batch-load relationships
            rels_map: dict[str, list[RelationshipClaim]] = {
                pid: [] for pid in person_ids
            }
            rel_result = await session.execute(
                text(
                    f"SELECT person_id, relationship, stated_by, created_at FROM person_relationships WHERE person_id IN ({placeholders})"
                ),
                id_params,
            )
            for rrow in rel_result.fetchall():
                rels_map[rrow[0]].append(
                    RelationshipClaim(
                        relationship=rrow[1],
                        stated_by=rrow[2],
                        created_at=_parse_datetime(rrow[3]),
                    )
                )

            people = []
            for row in rows:
                person = _row_to_person(row, aliases_map[row.id], rels_map[row.id])
                people.append(person)
            return people

    async def get_all_people(self: Store) -> list[PersonEntry]:
        async with self._db.session() as session:
            result = await session.execute(text("SELECT id FROM people"))
            people = []
            for row in result.fetchall():
                person = await load_person_full(session, row[0])
                if person:
                    people.append(person)
            return people

    async def update_person(
        self: Store,
        person_id: str,
        name: str | None = None,
        updated_by: str | None = None,
        clear_merged: bool = False,
    ) -> PersonEntry | None:
        now = datetime.now(UTC)
        async with self._db.session() as session:
            result = await session.execute(
                text("SELECT id FROM people WHERE id = :id"),
                {"id": person_id},
            )
            if not result.fetchone():
                return None

            updates = ["updated_at = :updated_at"]
            params: dict[str, Any] = {
                "id": person_id,
                "updated_at": now.isoformat(),
            }
            if name is not None:
                updates.append("name = :name")
                params["name"] = name
            if clear_merged:
                updates.append("merged_into = NULL")

            await session.execute(
                text(f"UPDATE people SET {', '.join(updates)} WHERE id = :id"),
                params,
            )

            return await load_person_full(session, person_id)

    async def get_person_names_batch(
        self: Store, person_ids: list[str]
    ) -> dict[str, str]:
        """Get names for multiple person IDs in a single query.

        Returns a dict mapping person_id -> name for found persons.
        Missing IDs are not included in the result.
        """
        if not person_ids:
            return {}

        async with self._db.session() as session:
            # SQLite doesn't support array parameters, so build the query
            placeholders = ", ".join(f":id{i}" for i in range(len(person_ids)))
            params = {f"id{i}": pid for i, pid in enumerate(person_ids)}
            result = await session.execute(
                text(f"SELECT id, name FROM people WHERE id IN ({placeholders})"),
                params,
            )
            return {row[0]: row[1] for row in result.fetchall()}

    async def delete_person(self: Store, person_id: str) -> bool:
        async with self._db.session() as session:
            result = await session.execute(
                text("SELECT id, name FROM people WHERE id = :id"),
                {"id": person_id},
            )
            row = result.fetchone()
            if not row:
                return False

            name = row[1]

            # Clear merged_into references
            await session.execute(
                text("UPDATE people SET merged_into = NULL WHERE merged_into = :id"),
                {"id": person_id},
            )
            # Delete aliases and relationships (cascade would handle it but be explicit)
            await session.execute(
                text("DELETE FROM person_aliases WHERE person_id = :id"),
                {"id": person_id},
            )
            await session.execute(
                text("DELETE FROM person_relationships WHERE person_id = :id"),
                {"id": person_id},
            )
            await session.execute(
                text("DELETE FROM people WHERE id = :id"),
                {"id": person_id},
            )

            logger.debug(
                "person_deleted",
                extra={"person_id": person_id, "person_name": name},
            )
            return True
