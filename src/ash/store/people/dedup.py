"""Person deduplication and merge operations."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import text

from ash.store.people.helpers import (
    load_person_full,
    normalize_reference,
    primary_sort_key,
)
from ash.store.types import PersonEntry

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)


class PeopleDedupMixin:
    """Person deduplication, merge, and chain-following operations."""

    async def merge_people(
        self: Store,
        primary_id: str,
        secondary_id: str,
    ) -> PersonEntry | None:
        async with self._db.session() as session:
            primary = await load_person_full(session, primary_id)
            secondary = await load_person_full(session, secondary_id)
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
                    new_is_self = any(
                        r.relationship.lower() == "self"
                        for r in new_person.relationships
                    )
                    existing_is_self = any(
                        r.relationship.lower() == "self" for r in existing.relationships
                    )
                    if new_is_self and existing_is_self:
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
            members.sort(key=primary_sort_key)
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
    def _heuristic_match(a: PersonEntry, b: PersonEntry) -> bool:
        # Normalized alias overlap (strong identity signal, checked first)
        a_aliases = {normalize_reference(alias.value) for alias in a.aliases}
        b_aliases = {normalize_reference(alias.value) for alias in b.aliases}
        a_aliases.discard("")
        b_aliases.discard("")
        if a_aliases & b_aliases:
            return True

        # Name-to-alias cross-matching
        a_name_norm = normalize_reference(a.name)
        b_name_norm = normalize_reference(b.name)
        if a_name_norm and a_name_norm in b_aliases:
            return True
        if b_name_norm and b_name_norm in a_aliases:
            return True

        # Self-check (after alias checks so shared aliases override)
        a_rels = {r.relationship.lower() for r in a.relationships}
        b_rels = {r.relationship.lower() for r in b.relationships}
        if "self" in a_rels and "self" in b_rels and a.created_by != b.created_by:
            return False

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
