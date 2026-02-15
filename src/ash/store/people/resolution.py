"""Person resolution and lookup operations."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from sqlalchemy import text

from ash.store.people.helpers import (
    ALIAS_NORM_MATCH,
    FUZZY_MATCH_PROMPT,
    RELATIONSHIP_TERMS,
    load_person_full,
    normalize_reference,
)
from ash.store.types import (
    PersonEntry,
    PersonResolutionResult,
)

if TYPE_CHECKING:
    from ash.store.store import Store

logger = logging.getLogger(__name__)


class PeopleResolutionMixin:
    """Person lookup and resolution operations."""

    async def find_person(self: Store, reference: str) -> PersonEntry | None:
        ref = normalize_reference(reference)
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
                return await load_person_full(session, row[0])

            # Search by alias - match normalized forms (strip my/the/@ prefixes)
            result = await session.execute(
                text(f"""
                    SELECT pa.person_id FROM person_aliases pa
                    JOIN people p ON p.id = pa.person_id
                    WHERE p.merged_into IS NULL AND {ALIAS_NORM_MATCH}
                """),
                {"ref": ref},
            )
            row = result.fetchone()
            if row:
                return await load_person_full(session, row[0])

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
                return await load_person_full(session, row[0])

        return None

    async def find_person_for_speaker(
        self: Store,
        reference: str,
        speaker_user_id: str,
    ) -> PersonEntry | None:
        ref = normalize_reference(reference)
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
                    WHERE p.merged_into IS NULL AND {ALIAS_NORM_MATCH}
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
                    return await load_person_full(session, pid)

                r = await session.execute(
                    text(
                        "SELECT COUNT(*) FROM person_aliases WHERE person_id = :pid AND added_by = :speaker"
                    ),
                    {"pid": pid, "speaker": speaker_user_id},
                )
                if (r.scalar() or 0) > 0:
                    return await load_person_full(session, pid)

        return None

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
