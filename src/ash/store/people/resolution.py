"""Person resolution and lookup operations."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from ash.store.people.helpers import (
    FUZZY_MATCH_PROMPT,
    RELATIONSHIP_TERMS,
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

    @staticmethod
    def _lexical_tokens(value: str) -> set[str]:
        """Tokenize for lightweight lexical overlap checks."""
        return {token for token in re.findall(r"[a-z0-9]+", value) if len(token) >= 3}

    @classmethod
    def _is_plausible_fuzzy_match(
        cls,
        reference: str,
        person: PersonEntry,
        *,
        content_hint: str | None = None,
    ) -> bool:
        """Safety gate for LLM-picked fuzzy person matches.

        LLM fuzzy matching can occasionally return a plausible-looking but wrong
        person ID. We require deterministic lexical evidence between the
        reference and the selected person before accepting the match.
        """
        ref_norm = normalize_reference(reference)
        if not ref_norm:
            return False

        ref_tokens = cls._lexical_tokens(ref_norm)
        best_ratio = 0.0

        candidate_values = [person.name]
        candidate_values.extend(alias.value for alias in person.aliases)
        candidate_values.extend(rel.relationship for rel in person.relationships)

        for value in candidate_values:
            cand_norm = normalize_reference(value)
            if not cand_norm:
                continue

            if cand_norm == ref_norm:
                return True

            compact = cand_norm.replace(" ", "")
            if len(ref_norm) >= 4 and (
                ref_norm in cand_norm
                or cand_norm in ref_norm
                or ref_norm in compact
                or compact in ref_norm
            ):
                return True

            ratio = SequenceMatcher(None, ref_norm, cand_norm).ratio()
            best_ratio = max(best_ratio, ratio)

            cand_tokens = cls._lexical_tokens(cand_norm)
            if ref_tokens & cand_tokens:
                return True
            if any(
                len(token) >= 5 and (token in ref_norm or ref_norm in token)
                for token in cand_tokens
            ):
                return True

        if best_ratio >= 0.78:
            return True

        # Relationship bridge fallback:
        # if the content explicitly contains the reference and a known
        # relationship term for this person, allow the match.
        if content_hint:
            content_norm = normalize_reference(content_hint)
            if ref_norm in content_norm:
                for rel in person.relationships:
                    rel_norm = normalize_reference(rel.relationship)
                    if rel_norm and rel_norm in content_norm:
                        return True

        return False

    async def find_person(self: Store, reference: str) -> PersonEntry | None:
        from ash.graph.edges import get_merged_into

        ref = normalize_reference(reference)

        for person in self._graph.people.values():
            if get_merged_into(self._graph, person.id) is not None:
                continue
            if person.name and person.name.lower() == ref:
                return person

        # Search by alias
        for person in self._graph.people.values():
            if get_merged_into(self._graph, person.id) is not None:
                continue
            for alias in person.aliases:
                alias_norm = normalize_reference(alias.value)
                if alias_norm == ref:
                    return person

        # Search by relationship
        for person in self._graph.people.values():
            if get_merged_into(self._graph, person.id) is not None:
                continue
            for rel in person.relationships:
                if rel.relationship.lower() == ref:
                    return person

        return None

    async def find_person_for_speaker(
        self: Store,
        reference: str,
        speaker_user_id: str,
    ) -> PersonEntry | None:
        from ash.graph.edges import get_merged_into

        ref = normalize_reference(reference)
        candidate_ids: set[str] = set()

        for person in self._graph.people.values():
            if get_merged_into(self._graph, person.id) is not None:
                continue
            if person.name and person.name.lower() == ref:
                candidate_ids.add(person.id)
            for alias in person.aliases:
                alias_norm = normalize_reference(alias.value)
                if alias_norm == ref:
                    candidate_ids.add(person.id)
            for rel in person.relationships:
                if rel.relationship.lower() == ref:
                    candidate_ids.add(person.id)

        for pid in candidate_ids:
            person = self._graph.people.get(pid)
            if not person:
                continue
            # Check if speaker is connected via relationships
            for rel in person.relationships:
                if rel.stated_by == speaker_user_id:
                    return person
            # Check if speaker is connected via aliases
            for alias in person.aliases:
                if alias.added_by == speaker_user_id:
                    return person

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
            if not await self._has_ambiguous_matches(reference):
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
        return await self.get_person_names_batch(person_ids)

    async def _fuzzy_find(
        self: Store,
        reference: str,
        content_hint: str | None = None,
        speaker: str | None = None,
    ) -> PersonEntry | None:
        if not self._llm or not self._llm_model:
            return None
        people = await self.get_all_people()
        from ash.graph.edges import get_merged_into

        candidates = [p for p in people if not get_merged_into(self._graph, p.id)]
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
            for p in candidates:
                if p.id == result:
                    if not self._is_plausible_fuzzy_match(
                        reference,
                        p,
                        content_hint=content_hint,
                    ):
                        logger.warning(
                            "fuzzy_match_rejected",
                            extra={
                                "reference": reference,
                                "candidate.person_id": p.id,
                                "candidate.person_name": p.name,
                            },
                        )
                        return None
                    return p
            return None
        except Exception:
            logger.warning("fuzzy_find_failed", exc_info=True)
            return None

    async def _has_ambiguous_matches(self: Store, reference: str) -> bool:
        """Check if a reference matches multiple distinct (non-merged) people."""
        from ash.graph.edges import get_merged_into

        ref = normalize_reference(reference)
        person_ids: set[str] = set()

        for person in self._graph.people.values():
            if get_merged_into(self._graph, person.id) is not None:
                continue
            if person.name and person.name.lower() == ref:
                person_ids.add(person.id)
            for alias in person.aliases:
                alias_norm = normalize_reference(alias.value)
                if alias_norm == ref:
                    person_ids.add(person.id)

        return len(person_ids) > 1

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
