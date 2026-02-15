"""Person alias and relationship management."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import text

from ash.store.people.helpers import load_person_full
from ash.store.types import PersonEntry

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
        async with self._db.session() as session:
            result = await session.execute(
                text("SELECT id FROM people WHERE id = :id"),
                {"id": person_id},
            )
            if not result.fetchone():
                return None

            # Check if alias already exists
            result = await session.execute(
                text(
                    "SELECT COUNT(*) FROM person_aliases WHERE person_id = :pid AND LOWER(value) = :val"
                ),
                {"pid": person_id, "val": alias.lower()},
            )
            if (result.scalar() or 0) == 0:
                now = datetime.now(UTC)
                await session.execute(
                    text("""
                        INSERT INTO person_aliases (person_id, value, added_by, created_at)
                        VALUES (:pid, :value, :added_by, :created_at)
                    """),
                    {
                        "pid": person_id,
                        "value": alias,
                        "added_by": added_by,
                        "created_at": now.isoformat(),
                    },
                )
                await session.execute(
                    text("UPDATE people SET updated_at = :now WHERE id = :id"),
                    {"now": now.isoformat(), "id": person_id},
                )

            return await load_person_full(session, person_id)

    async def add_relationship(
        self: Store,
        person_id: str,
        relationship: str,
        stated_by: str | None = None,
    ) -> PersonEntry | None:
        async with self._db.session() as session:
            result = await session.execute(
                text("SELECT id FROM people WHERE id = :id"),
                {"id": person_id},
            )
            if not result.fetchone():
                return None

            # Check if relationship already exists
            result = await session.execute(
                text(
                    "SELECT COUNT(*) FROM person_relationships WHERE person_id = :pid AND LOWER(relationship) = :rel"
                ),
                {"pid": person_id, "rel": relationship.lower()},
            )
            if (result.scalar() or 0) == 0:
                now = datetime.now(UTC)
                await session.execute(
                    text("""
                        INSERT INTO person_relationships (person_id, relationship, stated_by, created_at)
                        VALUES (:pid, :rel, :stated_by, :created_at)
                    """),
                    {
                        "pid": person_id,
                        "rel": relationship,
                        "stated_by": stated_by,
                        "created_at": now.isoformat(),
                    },
                )
                await session.execute(
                    text("UPDATE people SET updated_at = :now WHERE id = :id"),
                    {"now": now.isoformat(), "id": person_id},
                )

            return await load_person_full(session, person_id)
