"""Person CRUD, resolution, and dedup mixin for GraphStore."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ash.people.types import (
    AliasEntry,
    PersonEntry,
    PersonResolutionResult,
    RelationshipClaim,
)

if TYPE_CHECKING:
    from ash.graph.store import GraphStore

logger = logging.getLogger(__name__)

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


class PeopleOpsMixin:
    """Person CRUD, resolution, fuzzy matching, merge, and dedup."""

    async def create_person(
        self: GraphStore,
        created_by: str,
        name: str,
        relationship: str | None = None,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        relationship_stated_by: str | None = None,
    ) -> PersonEntry:
        now = datetime.now(UTC)
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
            id=str(uuid.uuid4()),
            version=1,
            created_by=created_by,
            name=name,
            relationships=relationships,
            aliases=alias_entries,
            created_at=now,
            updated_at=now,
            metadata=metadata,
        )
        await self._people_jsonl.append(entry)
        self._invalidate_people_cache()
        logger.debug(
            "person_created", extra={"person_id": entry.id, "person_name": name}
        )
        return entry

    async def get_person(self: GraphStore, person_id: str) -> PersonEntry | None:
        people = await self._ensure_people_loaded()
        return self._find_person_by_id(people, person_id)

    async def find_person(self: GraphStore, reference: str) -> PersonEntry | None:
        ref = self._normalize_reference(reference)
        people = await self._ensure_people_loaded()
        for person in people:
            if person.merged_into:
                continue
            if person.name.lower() == ref:
                return person
            for rc in person.relationships:
                if rc.relationship.lower() == ref:
                    return person
            for alias in person.aliases:
                if self._normalize_reference(alias.value) == ref:
                    return person
        return None

    async def find_person_for_speaker(
        self: GraphStore,
        reference: str,
        speaker_user_id: str,
    ) -> PersonEntry | None:
        ref = self._normalize_reference(reference)
        people = await self._ensure_people_loaded()
        for person in people:
            if person.merged_into:
                continue
            is_connected = any(
                rc.stated_by == speaker_user_id for rc in person.relationships
            ) or any(ae.added_by == speaker_user_id for ae in person.aliases)
            if not is_connected:
                continue
            if person.name.lower() == ref:
                return person
            for rc in person.relationships:
                if rc.relationship.lower() == ref:
                    return person
            for alias in person.aliases:
                if self._normalize_reference(alias.value) == ref:
                    return person
        return None

    async def list_people(self: GraphStore) -> list[PersonEntry]:
        people = await self._ensure_people_loaded()
        result = [p for p in people if not p.merged_into]
        result.sort(key=lambda x: x.name)
        return result

    async def get_all_people(self: GraphStore) -> list[PersonEntry]:
        return await self._ensure_people_loaded()

    async def update_person(
        self: GraphStore,
        person_id: str,
        name: str | None = None,
        relationship: str | None = None,
        aliases: list[str] | None = None,
        updated_by: str | None = None,
        clear_merged: bool = False,
    ) -> PersonEntry | None:
        people = await self._ensure_people_loaded()
        person = self._find_person_by_id(people, person_id)
        if not person:
            return None
        now = datetime.now(UTC)
        if name is not None:
            person.name = name
        if relationship is not None:
            person.relationships = [
                RelationshipClaim(
                    relationship=relationship, stated_by=updated_by, created_at=now
                )
            ]
        if aliases is not None:
            person.aliases = [
                AliasEntry(value=a, added_by=updated_by, created_at=now)
                for a in aliases
            ]
        if clear_merged:
            person.merged_into = None
        person.updated_at = now
        await self._people_jsonl.rewrite(people)
        self._invalidate_people_cache()
        return person

    async def add_alias(
        self: GraphStore,
        person_id: str,
        alias: str,
        added_by: str | None = None,
    ) -> PersonEntry | None:
        people = await self._ensure_people_loaded()
        person = self._find_person_by_id(people, person_id)
        if not person:
            return None
        existing_values = [a.value.lower() for a in person.aliases]
        if alias.lower() not in existing_values:
            person.aliases.append(
                AliasEntry(value=alias, added_by=added_by, created_at=datetime.now(UTC))
            )
            person.updated_at = datetime.now(UTC)
            await self._people_jsonl.rewrite(people)
            self._invalidate_people_cache()
        return person

    async def add_relationship(
        self: GraphStore,
        person_id: str,
        relationship: str,
        stated_by: str | None = None,
    ) -> PersonEntry | None:
        people = await self._ensure_people_loaded()
        person = self._find_person_by_id(people, person_id)
        if not person:
            return None
        existing_rels = [r.relationship.lower() for r in person.relationships]
        if relationship.lower() not in existing_rels:
            person.relationships.append(
                RelationshipClaim(
                    relationship=relationship,
                    stated_by=stated_by,
                    created_at=datetime.now(UTC),
                )
            )
            person.updated_at = datetime.now(UTC)
            await self._people_jsonl.rewrite(people)
            self._invalidate_people_cache()
        return person

    async def delete_person(self: GraphStore, person_id: str) -> bool:
        people = await self._ensure_people_loaded()
        person = self._find_person_by_id(people, person_id)
        if not person:
            return False
        people.remove(person)
        for p in people:
            if p.merged_into == person_id:
                p.merged_into = None
        await self._people_jsonl.rewrite(people)
        self._invalidate_people_cache()
        logger.debug(
            "person_deleted", extra={"person_id": person_id, "person_name": person.name}
        )
        return True

    async def merge_people(
        self: GraphStore,
        primary_id: str,
        secondary_id: str,
    ) -> PersonEntry | None:
        people = await self._ensure_people_loaded()
        primary = self._find_person_by_id(people, primary_id)
        secondary = self._find_person_by_id(people, secondary_id)
        if not primary or not secondary:
            return None
        if secondary.merged_into:
            logger.debug(
                "Skipping merge: secondary %s already merged into %s",
                secondary_id,
                secondary.merged_into,
            )
            return None

        existing_values = {a.value.lower() for a in primary.aliases}
        for alias in secondary.aliases:
            if alias.value.lower() not in existing_values:
                primary.aliases.append(alias)
                existing_values.add(alias.value.lower())
        if (
            secondary.name.lower() != primary.name.lower()
            and secondary.name.lower() not in existing_values
        ):
            primary.aliases.append(
                AliasEntry(
                    value=secondary.name, added_by=None, created_at=datetime.now(UTC)
                )
            )
        existing_rels = {r.relationship.lower() for r in primary.relationships}
        for rc in secondary.relationships:
            if rc.relationship.lower() not in existing_rels:
                primary.relationships.append(rc)
                existing_rels.add(rc.relationship.lower())

        secondary.merged_into = primary_id
        primary.updated_at = datetime.now(UTC)
        await self._people_jsonl.rewrite(people)
        self._invalidate_people_cache()

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

        return primary

    async def resolve_or_create_person(
        self: GraphStore,
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

    async def resolve_names(self: GraphStore, person_ids: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for pid in person_ids:
            person = await self.get_person(pid)
            if person:
                result[pid] = person.name
        return result

    async def find_dedup_candidates(
        self: GraphStore,
        person_ids: list[str],
        *,
        exclude_self: bool = False,
    ) -> list[tuple[str, str]]:
        if not self._llm or not self._llm_model:
            return []
        people = await self._ensure_people_loaded()
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

        # Collect LLM-verified pairs
        verified_pairs: list[tuple[PersonEntry, PersonEntry]] = []
        for person_a, person_b in candidates:
            if await self._llm_verify_same_person(person_a, person_b):
                verified_pairs.append((person_a, person_b))

        if not verified_pairs:
            return []

        # Cluster verified pairs with union-find so that 3+ duplicates
        # produce exactly N-1 merges into a single primary, rather than
        # redundant/conflicting pairwise merges.
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

        # Group into clusters
        clusters: dict[str, list[PersonEntry]] = {}
        for pid, person in person_by_id.items():
            root = find(pid)
            clusters.setdefault(root, []).append(person)

        # For each cluster, pick the best primary and emit N-1 merges.
        # Sort key matches _pick_primary: most data first, then earliest.
        _epoch = datetime.min.replace(tzinfo=UTC)

        def _sort_key(p: PersonEntry) -> tuple[int, datetime]:
            score = len(p.aliases) + len(p.relationships)
            return (-score, p.created_at or _epoch)

        results: list[tuple[str, str]] = []
        for members in clusters.values():
            if len(members) < 2:
                continue
            members.sort(key=_sort_key)
            primary = members[0]
            for secondary in members[1:]:
                results.append((primary.id, secondary.id))

        return results

    @staticmethod
    def _find_person_by_id(
        people: list[PersonEntry], person_id: str
    ) -> PersonEntry | None:
        for p in people:
            if p.id == person_id:
                return p
        return None

    async def _follow_merge_chain(self: GraphStore, person: PersonEntry) -> PersonEntry:
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
        if "self" in a_rels and "self" in b_rels:
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
        self: GraphStore, a: PersonEntry, b: PersonEntry
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
    def _pick_primary(a: PersonEntry, b: PersonEntry) -> tuple[str, str]:
        _epoch = datetime.min.replace(tzinfo=UTC)

        def _sort_key(p: PersonEntry) -> tuple[int, datetime]:
            score = len(p.aliases) + len(p.relationships)
            return (-score, p.created_at or _epoch)

        first, second = sorted([a, b], key=_sort_key)
        return first.id, second.id

    async def _fuzzy_find(
        self: GraphStore,
        reference: str,
        content_hint: str | None = None,
        speaker: str | None = None,
    ) -> PersonEntry | None:
        if not self._llm or not self._llm_model:
            return None
        people = await self._ensure_people_loaded()
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
            return self._find_person_by_id(candidates, result)
        except Exception:
            logger.warning("fuzzy_find_failed", exc_info=True)
            return None

    def _parse_person_reference(
        self: GraphStore,
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
        return ref_lower.title(), None

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
