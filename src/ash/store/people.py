"""Person CRUD, resolution, and dedup mixin for Store (SQLite-backed)."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from ash.store.mappers import row_to_person as _row_to_person
from ash.store.types import (
    AliasEntry,
    PersonEntry,
    PersonResolutionResult,
    RelationshipClaim,
    _parse_datetime,
)

# Sentinel for sort key when created_at is None
_EPOCH = datetime.min.replace(tzinfo=UTC)

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)

# SQL clause to normalize alias values the same way as _normalize_reference:
# lowercase, trim, then strip 'my ', 'the ', '@' prefixes.
_ALIAS_NORM_MATCH = """(LOWER(TRIM(pa.value)) = :ref
    OR (LOWER(TRIM(pa.value)) LIKE 'my %' AND SUBSTR(LOWER(TRIM(pa.value)), 4) = :ref)
    OR (LOWER(TRIM(pa.value)) LIKE 'the %' AND SUBSTR(LOWER(TRIM(pa.value)), 5) = :ref)
    OR (LOWER(TRIM(pa.value)) LIKE '@%' AND SUBSTR(LOWER(TRIM(pa.value)), 2) = :ref))"""

FUZZY_MATCH_PROMPT = """Given a person reference and a list of known people, determine if the reference matches any existing person.

Reference: "{reference}"
{context_section}
{speaker_section}
Known people:
{people_list}

Consider: name variants (first name <-> full name, nicknames), relationship links (e.g., "Sarah" from speaker "dcramer" matches a person with relationship "wife" stated by "dcramer"), and alias matches. Prefer matching relationships stated by the current speaker.

If the reference clearly refers to one of the known people, respond with ONLY the ID.
If no clear match, respond with NONE.

Respond with only the ID or NONE, nothing else."""

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


async def _load_person_full(session, person_id: str) -> PersonEntry | None:
    """Load a person with aliases and relationships."""
    result = await session.execute(
        text("SELECT * FROM people WHERE id = :id"),
        {"id": person_id},
    )
    row = result.fetchone()
    if not row:
        return None

    aliases = await _load_aliases(session, person_id)
    relationships = await _load_relationships(session, person_id)
    return _row_to_person(row, aliases, relationships)


async def _load_aliases(session, person_id: str) -> list[AliasEntry]:
    result = await session.execute(
        text(
            "SELECT value, added_by, created_at FROM person_aliases WHERE person_id = :id"
        ),
        {"id": person_id},
    )
    return [
        AliasEntry(
            value=row[0],
            added_by=row[1],
            created_at=_parse_datetime(row[2]),
        )
        for row in result.fetchall()
    ]


async def _load_relationships(session, person_id: str) -> list[RelationshipClaim]:
    result = await session.execute(
        text(
            "SELECT relationship, stated_by, created_at FROM person_relationships WHERE person_id = :id"
        ),
        {"id": person_id},
    )
    return [
        RelationshipClaim(
            relationship=row[0],
            stated_by=row[1],
            created_at=_parse_datetime(row[2]),
        )
        for row in result.fetchall()
    ]


class PeopleOpsMixin:
    """Person CRUD, resolution, fuzzy matching, merge, and dedup."""

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
            return await _load_person_full(session, person_id)

    async def find_person(self: Store, reference: str) -> PersonEntry | None:
        ref = self._normalize_reference(reference)
        async with self._db.session() as session:
            # Search by name
            result = await session.execute(
                text(
                    "SELECT id FROM people WHERE LOWER(name) = :ref AND merged_into IS NULL"
                ),
                {"ref": ref},
            )
            row = result.fetchone()
            if row:
                return await _load_person_full(session, row[0])

            # Search by alias — match normalized forms (strip my/the/@ prefixes)
            result = await session.execute(
                text(f"""
                    SELECT pa.person_id FROM person_aliases pa
                    JOIN people p ON p.id = pa.person_id
                    WHERE p.merged_into IS NULL AND {_ALIAS_NORM_MATCH}
                """),
                {"ref": ref},
            )
            row = result.fetchone()
            if row:
                return await _load_person_full(session, row[0])

            # Search by relationship
            result = await session.execute(
                text("""
                    SELECT pr.person_id FROM person_relationships pr
                    JOIN people p ON p.id = pr.person_id
                    WHERE LOWER(pr.relationship) = :ref AND p.merged_into IS NULL
                """),
                {"ref": ref},
            )
            row = result.fetchone()
            if row:
                return await _load_person_full(session, row[0])

        return None

    async def find_person_for_speaker(
        self: Store,
        reference: str,
        speaker_user_id: str,
    ) -> PersonEntry | None:
        ref = self._normalize_reference(reference)
        async with self._db.session() as session:
            # Find by name or alias, then check speaker connection
            candidate_ids: set[str] = set()

            result = await session.execute(
                text(
                    "SELECT id FROM people WHERE LOWER(name) = :ref AND merged_into IS NULL"
                ),
                {"ref": ref},
            )
            for row in result.fetchall():
                candidate_ids.add(row[0])

            result = await session.execute(
                text(f"""
                    SELECT pa.person_id FROM person_aliases pa
                    JOIN people p ON p.id = pa.person_id
                    WHERE p.merged_into IS NULL AND {_ALIAS_NORM_MATCH}
                """),
                {"ref": ref},
            )
            for row in result.fetchall():
                candidate_ids.add(row[0])

            result = await session.execute(
                text("""
                    SELECT pr.person_id FROM person_relationships pr
                    JOIN people p ON p.id = pr.person_id
                    WHERE LOWER(pr.relationship) = :ref AND p.merged_into IS NULL
                """),
                {"ref": ref},
            )
            for row in result.fetchall():
                candidate_ids.add(row[0])

            for pid in candidate_ids:
                # Check if speaker is connected
                r = await session.execute(
                    text(
                        "SELECT COUNT(*) FROM person_relationships WHERE person_id = :pid AND stated_by = :speaker"
                    ),
                    {"pid": pid, "speaker": speaker_user_id},
                )
                if (r.scalar() or 0) > 0:
                    return await _load_person_full(session, pid)

                r = await session.execute(
                    text(
                        "SELECT COUNT(*) FROM person_aliases WHERE person_id = :pid AND added_by = :speaker"
                    ),
                    {"pid": pid, "speaker": speaker_user_id},
                )
                if (r.scalar() or 0) > 0:
                    return await _load_person_full(session, pid)

        return None

    async def list_people(self: Store) -> list[PersonEntry]:
        async with self._db.session() as session:
            result = await session.execute(
                text("SELECT id FROM people WHERE merged_into IS NULL ORDER BY name"),
            )
            people = []
            for row in result.fetchall():
                person = await _load_person_full(session, row[0])
                if person:
                    people.append(person)
            return people

    async def get_all_people(self: Store) -> list[PersonEntry]:
        async with self._db.session() as session:
            result = await session.execute(text("SELECT id FROM people"))
            people = []
            for row in result.fetchall():
                person = await _load_person_full(session, row[0])
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

            return await _load_person_full(session, person_id)

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

            return await _load_person_full(session, person_id)

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

            return await _load_person_full(session, person_id)

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

    async def merge_people(
        self: Store,
        primary_id: str,
        secondary_id: str,
    ) -> PersonEntry | None:
        async with self._db.session() as session:
            primary = await _load_person_full(session, primary_id)
            secondary = await _load_person_full(session, secondary_id)
            if not primary or not secondary:
                return None
            if secondary.merged_into:
                logger.debug(
                    "Skipping merge: secondary %s already merged into %s",
                    secondary_id,
                    secondary.merged_into,
                )
                return None

            now = datetime.now(UTC)

            # Merge aliases
            existing_values = {a.value.lower() for a in primary.aliases}
            for alias in secondary.aliases:
                if alias.value.lower() not in existing_values:
                    await session.execute(
                        text("""
                            INSERT INTO person_aliases (person_id, value, added_by, created_at)
                            VALUES (:pid, :value, :added_by, :created_at)
                        """),
                        {
                            "pid": primary_id,
                            "value": alias.value,
                            "added_by": alias.added_by,
                            "created_at": alias.created_at.isoformat()
                            if alias.created_at
                            else None,
                        },
                    )
                    existing_values.add(alias.value.lower())

            # Add secondary name as alias if different
            if (
                secondary.name.lower() != primary.name.lower()
                and secondary.name.lower() not in existing_values
            ):
                await session.execute(
                    text("""
                        INSERT INTO person_aliases (person_id, value, added_by, created_at)
                        VALUES (:pid, :value, NULL, :created_at)
                    """),
                    {
                        "pid": primary_id,
                        "value": secondary.name,
                        "created_at": now.isoformat(),
                    },
                )

            # Merge relationships
            existing_rels = {r.relationship.lower() for r in primary.relationships}
            for rc in secondary.relationships:
                if rc.relationship.lower() not in existing_rels:
                    await session.execute(
                        text("""
                            INSERT INTO person_relationships (person_id, relationship, stated_by, created_at)
                            VALUES (:pid, :rel, :stated_by, :created_at)
                        """),
                        {
                            "pid": primary_id,
                            "rel": rc.relationship,
                            "stated_by": rc.stated_by,
                            "created_at": rc.created_at.isoformat()
                            if rc.created_at
                            else None,
                        },
                    )
                    existing_rels.add(rc.relationship.lower())

            # Mark secondary as merged
            await session.execute(
                text(
                    "UPDATE people SET merged_into = :primary_id WHERE id = :secondary_id"
                ),
                {"primary_id": primary_id, "secondary_id": secondary_id},
            )
            await session.execute(
                text("UPDATE people SET updated_at = :now WHERE id = :id"),
                {"now": now.isoformat(), "id": primary_id},
            )

        logger.debug(
            "person_merged",
            extra={"primary_id": primary_id, "secondary_id": secondary_id},
        )

        # Auto-remap memory references
        try:
            remapped = await self.remap_subject_person_id(secondary_id, primary_id)
            if remapped:
                logger.debug(
                    "Remapped %d memories from %s to %s",
                    remapped,
                    secondary_id,
                    primary_id,
                )
        except Exception:
            logger.warning("Failed to remap memories after merge", exc_info=True)

        return await self.get_person(primary_id)

    async def resolve_or_create_person(
        self: Store,
        created_by: str,
        reference: str,
        content_hint: str | None = None,
        relationship_stated_by: str | None = None,
    ) -> PersonResolutionResult:
        # Speaker-scoped search first
        if relationship_stated_by:
            speaker_match = await self.find_person_for_speaker(
                reference, relationship_stated_by
            )
            if speaker_match:
                person = await self._follow_merge_chain(speaker_match)
                return PersonResolutionResult(
                    person_id=person.id, created=False, person_name=person.name
                )

        existing = await self.find_person(reference)
        if existing:
            person = await self._follow_merge_chain(existing)
            return PersonResolutionResult(
                person_id=person.id, created=False, person_name=person.name
            )

        # Try fuzzy match
        fuzzy_match = await self._fuzzy_find(
            reference,
            content_hint=content_hint,
            speaker=relationship_stated_by or created_by,
        )
        if fuzzy_match:
            person = await self._follow_merge_chain(fuzzy_match)
            await self.add_alias(person.id, reference, added_by=created_by)
            logger.debug(
                "fuzzy_match_resolved",
                extra={
                    "reference": reference,
                    "person_id": person.id,
                    "person_name": person.name,
                },
            )
            return PersonResolutionResult(
                person_id=person.id, created=False, person_name=person.name
            )

        name, relationship = self._parse_person_reference(reference, content_hint)
        person = await self.create_person(
            created_by=created_by,
            name=name,
            relationship=relationship,
            aliases=[reference] if reference.lower() != name.lower() else None,
            relationship_stated_by=relationship_stated_by,
        )
        return PersonResolutionResult(
            person_id=person.id, created=True, person_name=person.name
        )

    async def resolve_names(self: Store, person_ids: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for pid in person_ids:
            person = await self.get_person(pid)
            if person:
                result[pid] = person.name
        return result

    async def find_dedup_candidates(
        self: Store,
        person_ids: list[str],
        *,
        exclude_self: bool = False,
    ) -> list[tuple[str, str]]:
        if not self._llm or not self._llm_model:
            return []
        people = await self.get_all_people()
        active = [p for p in people if not p.merged_into]
        new_people = [p for p in active if p.id in set(person_ids)]
        if not new_people:
            return []

        seen: set[frozenset[str]] = set()
        candidates: list[tuple[PersonEntry, PersonEntry]] = []
        for new_person in new_people:
            for existing in active:
                if existing.id == new_person.id:
                    continue
                pair_key = frozenset({new_person.id, existing.id})
                if pair_key in seen:
                    continue
                if exclude_self:
                    has_self = any(
                        r.relationship.lower() == "self"
                        for r in (*new_person.relationships, *existing.relationships)
                    )
                    if has_self:
                        continue
                if self._heuristic_match(new_person, existing):
                    seen.add(pair_key)
                    candidates.append((new_person, existing))

        if not candidates:
            return []

        verified_pairs: list[tuple[PersonEntry, PersonEntry]] = []
        for person_a, person_b in candidates:
            if await self._llm_verify_same_person(person_a, person_b):
                verified_pairs.append((person_a, person_b))

        if not verified_pairs:
            return []

        # Union-find clustering
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        person_by_id: dict[str, PersonEntry] = {}
        for a, b in verified_pairs:
            person_by_id[a.id] = a
            person_by_id[b.id] = b
            union(a.id, b.id)

        clusters: dict[str, list[PersonEntry]] = {}
        for pid, person in person_by_id.items():
            root = find(pid)
            clusters.setdefault(root, []).append(person)

        results: list[tuple[str, str]] = []
        for members in clusters.values():
            if len(members) < 2:
                continue
            members.sort(key=self._primary_sort_key)
            primary = members[0]
            for secondary in members[1:]:
                results.append((primary.id, secondary.id))

        return results

    async def _follow_merge_chain(self: Store, person: PersonEntry) -> PersonEntry:
        visited: set[str] = set()
        current = person
        while current.merged_into and current.merged_into not in visited:
            visited.add(current.id)
            next_person = await self.get_person(current.merged_into)
            if not next_person:
                break
            current = next_person
        return current

    @staticmethod
    def _normalize_reference(text: str) -> str:
        result = text.lower().strip()
        for prefix in ("my ", "the ", "@"):
            result = result.removeprefix(prefix)
        return result

    @staticmethod
    def _heuristic_match(a: PersonEntry, b: PersonEntry) -> bool:
        a_rels = {r.relationship.lower() for r in a.relationships}
        b_rels = {r.relationship.lower() for r in b.relationships}
        if "self" in a_rels and "self" in b_rels and a.created_by != b.created_by:
            return False
        a_aliases = {alias.value.lower() for alias in a.aliases}
        b_aliases = {alias.value.lower() for alias in b.aliases}
        if a_aliases & b_aliases:
            return True
        a_name = a.name.lower()
        b_name = b.name.lower()
        if a_name in b_rels or b_name in a_rels:
            return True
        if len(a_name) >= 3 and len(b_name) >= 3:
            if a_name in b_name or b_name in a_name:
                return True
        a_parts = set(a_name.split())
        b_parts = set(b_name.split())
        if (len(a_parts) == 1 or len(b_parts) == 1) and a_parts & b_parts:
            return True
        return False

    async def _llm_verify_same_person(
        self: Store, a: PersonEntry, b: PersonEntry
    ) -> bool:
        if not self._llm or not self._llm_model:
            return False
        try:
            from ash.llm.types import Message, Role

            a_aliases = ", ".join(al.value for al in a.aliases) or "none"
            b_aliases = ", ".join(al.value for al in b.aliases) or "none"
            a_rels = ", ".join(r.relationship for r in a.relationships) or "none"
            b_rels = ", ".join(r.relationship for r in b.relationships) or "none"
            prompt = (
                "Do these two person records refer to the same real-world person?\n\n"
                f"Person A: Name: {a.name}, Aliases: {a_aliases}, Relationships: {a_rels}\n"
                f"Person B: Name: {b.name}, Aliases: {b_aliases}, Relationships: {b_rels}\n\n"
                "Answer only YES or NO."
            )
            response = await self._llm.complete(
                messages=[Message(role=Role.USER, content=prompt)],
                model=self._llm_model,
                max_tokens=10,
                temperature=0.0,
            )
            return response.message.get_text().strip().upper() == "YES"
        except Exception:
            logger.warning("llm_verify_same_person_failed", exc_info=True)
            return False

    @staticmethod
    def _primary_sort_key(p: PersonEntry) -> tuple[int, datetime]:
        score = len(p.aliases) + len(p.relationships)
        return (-score, p.created_at or _EPOCH)

    @staticmethod
    def _pick_primary(a: PersonEntry, b: PersonEntry) -> tuple[str, str]:
        first, second = sorted([a, b], key=PeopleOpsMixin._primary_sort_key)
        return first.id, second.id

    async def _fuzzy_find(
        self: Store,
        reference: str,
        content_hint: str | None = None,
        speaker: str | None = None,
    ) -> PersonEntry | None:
        if not self._llm or not self._llm_model:
            return None
        people = await self.get_all_people()
        candidates = [p for p in people if not p.merged_into]
        if not candidates:
            return None
        try:
            from ash.llm.types import Message, Role

            lines = []
            for p in candidates:
                parts = [f"ID: {p.id}, Name: {p.name}"]
                if p.aliases:
                    alias_strs = [a.value for a in p.aliases]
                    parts.append(f"Aliases: {', '.join(alias_strs)}")
                if p.relationships:
                    rel_parts = []
                    for r in p.relationships:
                        if r.stated_by:
                            rel_parts.append(
                                f"{r.relationship} (stated by {r.stated_by})"
                            )
                        else:
                            rel_parts.append(r.relationship)
                    parts.append(f"Relationships: {', '.join(rel_parts)}")
                lines.append(" | ".join(parts))

            people_list = "\n".join(lines)
            context_section = f'Context: "{content_hint}"' if content_hint else ""
            speaker_section = f'Speaker: "{speaker}"' if speaker else ""
            prompt = FUZZY_MATCH_PROMPT.format(
                reference=reference,
                people_list=people_list,
                context_section=context_section,
                speaker_section=speaker_section,
            )
            response = await self._llm.complete(
                messages=[Message(role=Role.USER, content=prompt)],
                model=self._llm_model,
                max_tokens=50,
                temperature=0.0,
            )
            result = response.message.get_text().strip()
            if result == "NONE":
                return None
            # Find the person by ID from candidates
            for p in candidates:
                if p.id == result:
                    return p
            return None
        except Exception:
            logger.warning("fuzzy_find_failed", exc_info=True)
            return None

    def _parse_person_reference(
        self: Store,
        reference: str,
        content_hint: str | None = None,
    ) -> tuple[str, str | None]:
        ref_lower = reference.lower().strip().removeprefix("@")
        relationship = (
            ref_lower.removeprefix("my ") if ref_lower.startswith("my ") else None
        )
        if relationship and relationship in RELATIONSHIP_TERMS:
            if content_hint:
                name = self._extract_name_from_content(content_hint, relationship)
                if name:
                    return name, relationship
            return relationship.title(), relationship
        cleaned = re.sub(r"\d+$", "", ref_lower).replace("_", " ").strip()
        return (cleaned or ref_lower).title(), None

    @staticmethod
    def _extract_name_from_content(content: str, relationship: str) -> str | None:
        def _extract_capitalized_name(text: str) -> str | None:
            match = re.match(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
            return match.group(1) if match else None

        match = re.search(
            rf"{relationship}(?:'s name is| is named)\s+",
            content,
            re.IGNORECASE,
        )
        if match:
            name = _extract_capitalized_name(content[match.end() :])
            if name:
                return name

        match = re.search(
            rf"(?:^|,\s*)my {relationship}\s+",
            content,
            re.IGNORECASE,
        )
        if match:
            name = _extract_capitalized_name(content[match.end() :])
            if name:
                return name

        match = re.search(r"^(\w+)'s\s", content)
        if match:
            name = match.group(1)
            if name.lower() not in ["user", "my", "the", "their", "his", "her"]:
                return name
        return None
