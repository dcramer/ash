"""Person management facade with storage.

Handles person entity CRUD operations, resolution, and persistence.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ash.config.paths import get_people_jsonl_path
from ash.memory.jsonl import TypedJSONL
from ash.people.types import (
    AliasEntry,
    PersonEntry,
    PersonResolutionResult,
    RelationshipClaim,
)

if TYPE_CHECKING:
    from ash.llm import LLMProvider
    from ash.memory import MemoryManager

logger = logging.getLogger(__name__)

FUZZY_MATCH_PROMPT = """Given a person reference and a list of known people, determine if the reference matches any existing person.

Reference: "{reference}"
{context_section}
{speaker_section}
Known people:
{people_list}

Consider: name variants (first name ↔ full name, nicknames), relationship links (e.g., "Sarah" from speaker "dcramer" matches a person with relationship "wife" stated by "dcramer"), and alias matches. Prefer matching relationships stated by the current speaker.

If the reference clearly refers to one of the known people, respond with ONLY the ID.
If no clear match, respond with NONE.

Respond with only the ID or NONE, nothing else."""


def _extract_capitalized_name(text: str) -> str | None:
    """Extract consecutive capitalized words from the start of text.

    "Sarah Jane loves hiking" -> "Sarah Jane"
    "Sarah loves hiking" -> "Sarah"
    "loves hiking" -> None
    """
    match = re.match(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
    return match.group(1) if match else None


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


class PersonManager:
    """Manages person entities with JSONL storage.

    Provides CRUD, resolution, and username matching for person records.
    People are global (no owner scoping) — created_by tracks who created
    the record, not who "owns" it.
    """

    def __init__(self, people_path: Path | None = None) -> None:
        self._people_jsonl = TypedJSONL(
            people_path or get_people_jsonl_path(), PersonEntry
        )

        # In-memory cache
        self._people_cache: list[PersonEntry] | None = None
        self._people_mtime: float | None = None

        # Optional LLM for fuzzy matching
        self._llm: LLMProvider | None = None
        self._llm_model: str | None = None

        # Optional memory manager for auto-remap on merge
        self._memory_manager: MemoryManager | None = None

    def set_llm(self, llm: LLMProvider, model: str) -> None:
        """Set LLM provider for fuzzy person matching.

        Called post-construction once the extraction LLM is available.
        """
        self._llm = llm
        self._llm_model = model

    def set_memory_manager(self, memory_manager: MemoryManager) -> None:
        """Set memory manager for auto-remap on merge.

        Called post-construction once the memory manager is available.
        """
        self._memory_manager = memory_manager

    async def _ensure_loaded(self) -> list[PersonEntry]:
        """Load people from disk if cache is stale."""
        current_mtime = (
            self._people_jsonl.get_mtime() if self._people_jsonl.exists() else None
        )

        if self._people_cache is None or current_mtime != self._people_mtime:
            self._people_cache = await self._people_jsonl.load_all()
            self._people_mtime = current_mtime

        return self._people_cache

    def _invalidate_cache(self) -> None:
        """Invalidate cache after write."""
        self._people_cache = None
        self._people_mtime = None

    # CRUD

    async def create(
        self,
        created_by: str,
        name: str,
        relationship: str | None = None,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        relationship_stated_by: str | None = None,
    ) -> PersonEntry:
        """Create a new person entity.

        Args:
            created_by: Who is creating this record.
            name: Primary display name.
            relationship: Optional relationship term (creates a RelationshipClaim).
            aliases: Optional alias strings (creates AliasEntry with added_by=created_by).
            metadata: Optional extra data.
            relationship_stated_by: Who stated the relationship (defaults to created_by).
                Use this when created_by is a numeric ID but the relationship was
                stated by a username.
        """
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
        self._invalidate_cache()

        logger.debug(
            "person_created", extra={"person_id": entry.id, "person_name": name}
        )
        return entry

    async def get(
        self,
        person_id: str,
    ) -> PersonEntry | None:
        """Get person by ID."""
        people = await self._ensure_loaded()
        return self._find_by_id(people, person_id)

    async def find(
        self,
        reference: str,
    ) -> PersonEntry | None:
        """Find person by name, relationship, or alias.

        Searches all records globally (no owner scoping).
        Excludes merged records.
        """
        ref = self._normalize_reference(reference)
        people = await self._ensure_loaded()

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

    async def find_for_speaker(
        self,
        reference: str,
        speaker_user_id: str,
    ) -> PersonEntry | None:
        """Find person by traversing the speaker's KNOWS/ALIAS edges first.

        Resolution starts at the speaker's graph neighborhood: persons where
        the speaker has a RelationshipClaim (KNOWS edge) or AliasEntry (ALIAS
        edge). If no match is found among connected persons, returns None to
        allow fallback to global find().

        Args:
            reference: Person reference to match (e.g., "Sarah", "wife").
            speaker_user_id: The user ID (or username) of the speaker.

        Returns:
            Matching person from speaker's connections, or None.
        """
        ref = self._normalize_reference(reference)
        people = await self._ensure_loaded()

        for person in people:
            if person.merged_into:
                continue

            # Check if speaker is connected to this person via KNOWS or ALIAS edges
            is_connected = any(
                rc.stated_by == speaker_user_id for rc in person.relationships
            ) or any(ae.added_by == speaker_user_id for ae in person.aliases)

            if not is_connected:
                continue

            # Match reference against connected person's attributes
            if person.name.lower() == ref:
                return person
            for rc in person.relationships:
                if rc.relationship.lower() == ref:
                    return person
            for alias in person.aliases:
                if self._normalize_reference(alias.value) == ref:
                    return person

        return None

    async def list_all(self) -> list[PersonEntry]:
        """Get all non-merged people, sorted by name."""
        people = await self._ensure_loaded()
        result = [p for p in people if not p.merged_into]
        result.sort(key=lambda x: x.name)
        return result

    async def update(
        self,
        person_id: str,
        name: str | None = None,
        relationship: str | None = None,
        aliases: list[str] | None = None,
        updated_by: str | None = None,
    ) -> PersonEntry | None:
        """Update person details.

        Args:
            person_id: Person to update.
            name: New display name.
            relationship: New relationship (replaces all existing).
            aliases: New aliases (replaces all existing).
            updated_by: Who is making this update (for provenance).
        """
        people = await self._ensure_loaded()
        person = self._find_by_id(people, person_id)

        if not person:
            return None

        now = datetime.now(UTC)
        if name is not None:
            person.name = name
        if relationship is not None:
            person.relationships = [
                RelationshipClaim(
                    relationship=relationship,
                    stated_by=updated_by,
                    created_at=now,
                )
            ]
        if aliases is not None:
            person.aliases = [
                AliasEntry(value=a, added_by=updated_by, created_at=now)
                for a in aliases
            ]
        person.updated_at = now

        await self._people_jsonl.rewrite(people)
        self._invalidate_cache()

        return person

    async def add_alias(
        self,
        person_id: str,
        alias: str,
        added_by: str | None = None,
    ) -> PersonEntry | None:
        """Add an alias to a person.

        Args:
            person_id: Person to add alias to.
            alias: The alias string.
            added_by: Who is adding this alias (for provenance).
        """
        people = await self._ensure_loaded()
        person = self._find_by_id(people, person_id)

        if not person:
            return None

        existing_values = [a.value.lower() for a in person.aliases]
        if alias.lower() not in existing_values:
            person.aliases.append(
                AliasEntry(
                    value=alias,
                    added_by=added_by,
                    created_at=datetime.now(UTC),
                )
            )
            person.updated_at = datetime.now(UTC)

            await self._people_jsonl.rewrite(people)
            self._invalidate_cache()

        return person

    async def add_relationship(
        self,
        person_id: str,
        relationship: str,
        stated_by: str | None = None,
    ) -> PersonEntry | None:
        """Add a relationship claim to a person.

        Args:
            person_id: Person to add relationship to.
            relationship: The relationship term (e.g., "wife").
            stated_by: Who stated this relationship.
        """
        people = await self._ensure_loaded()
        person = self._find_by_id(people, person_id)

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
            self._invalidate_cache()

        return person

    async def delete(self, person_id: str) -> bool:
        """Delete a person by ID.

        Also clears any merged_into references pointing to this person.

        Returns:
            True if the person was found and deleted, False otherwise.
        """
        people = await self._ensure_loaded()
        person = self._find_by_id(people, person_id)

        if not person:
            return False

        people.remove(person)

        # Clear any merged_into references pointing to the deleted person
        for p in people:
            if p.merged_into == person_id:
                p.merged_into = None

        await self._people_jsonl.rewrite(people)
        self._invalidate_cache()

        logger.debug(
            "person_deleted", extra={"person_id": person_id, "person_name": person.name}
        )
        return True

    async def get_all(self) -> list[PersonEntry]:
        """Get all people across all owners (including merged)."""
        return await self._ensure_loaded()

    # Merge

    async def merge(
        self,
        primary_id: str,
        secondary_id: str,
    ) -> PersonEntry | None:
        """Merge two person records.

        The secondary person is merged into the primary. Secondary's aliases
        and relationships are copied to primary, and secondary is marked
        with merged_into pointing to primary.

        Args:
            primary_id: The person to keep.
            secondary_id: The person to merge into primary.

        Returns:
            Updated primary person, or None if either ID not found.
        """
        people = await self._ensure_loaded()
        primary = self._find_by_id(people, primary_id)
        secondary = self._find_by_id(people, secondary_id)

        if not primary or not secondary:
            return None

        # Refuse to merge an already-merged secondary (prevents chain corruption)
        if secondary.merged_into:
            logger.debug(
                "Skipping merge: secondary %s already merged into %s",
                secondary_id,
                secondary.merged_into,
            )
            return None

        # Merge aliases (deduped, case-insensitive)
        existing_values = {a.value.lower() for a in primary.aliases}
        for alias in secondary.aliases:
            if alias.value.lower() not in existing_values:
                primary.aliases.append(alias)
                existing_values.add(alias.value.lower())

        # Add secondary's name as alias if different
        if (
            secondary.name.lower() != primary.name.lower()
            and secondary.name.lower() not in existing_values
        ):
            primary.aliases.append(
                AliasEntry(
                    value=secondary.name, added_by=None, created_at=datetime.now(UTC)
                )
            )

        # Copy relationships from secondary that primary doesn't have
        existing_rels = {r.relationship.lower() for r in primary.relationships}
        for rc in secondary.relationships:
            if rc.relationship.lower() not in existing_rels:
                primary.relationships.append(rc)
                existing_rels.add(rc.relationship.lower())

        # Mark secondary as merged
        secondary.merged_into = primary_id
        primary.updated_at = datetime.now(UTC)

        await self._people_jsonl.rewrite(people)
        self._invalidate_cache()

        logger.debug(
            "person_merged",
            extra={"primary_id": primary_id, "secondary_id": secondary_id},
        )

        # Auto-remap memory references from secondary to primary
        if self._memory_manager:
            try:
                remapped = await self._memory_manager.remap_subject_person_id(
                    secondary_id, primary_id
                )
                if remapped:
                    logger.debug(
                        "Remapped %d memories from %s to %s",
                        remapped,
                        secondary_id,
                        primary_id,
                    )
            except Exception:
                logger.debug("Failed to remap memories after merge", exc_info=True)

        return primary

    # Resolution

    async def resolve_or_create(
        self,
        created_by: str,
        reference: str,
        content_hint: str | None = None,
        relationship_stated_by: str | None = None,
    ) -> PersonResolutionResult:
        """Resolve a reference to a person, creating if needed.

        Resolution order:
        1. Exact match via find() (fast, deterministic)
        2. Fuzzy match via _fuzzy_find() (LLM fallback, optional)
        3. On fuzzy hit, add alias so exact match handles it next time
        4. On miss, create new person

        Args:
            created_by: Who is creating this record.
            reference: Person reference to resolve.
            content_hint: Optional content for name extraction.
            relationship_stated_by: Who stated the relationship (defaults to created_by).
        """
        # Speaker-scoped search first (graph traversal via KNOWS/ALIAS edges)
        if relationship_stated_by:
            speaker_match = await self.find_for_speaker(
                reference, relationship_stated_by
            )
            if speaker_match:
                person = await self._follow_merge_chain(speaker_match)
                return PersonResolutionResult(
                    person_id=person.id,
                    created=False,
                    person_name=person.name,
                )

        existing = await self.find(reference)
        if existing:
            # Follow merge chain
            person = await self._follow_merge_chain(existing)
            return PersonResolutionResult(
                person_id=person.id,
                created=False,
                person_name=person.name,
            )

        # Try fuzzy match before creating
        fuzzy_match = await self._fuzzy_find(
            reference,
            content_hint=content_hint,
            speaker=relationship_stated_by or created_by,
        )
        if fuzzy_match:
            person = await self._follow_merge_chain(fuzzy_match)
            # Learn this reference as an alias for next time
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
                person_id=person.id,
                created=False,
                person_name=person.name,
            )

        name, relationship = self._parse_person_reference(reference, content_hint)

        person = await self.create(
            created_by=created_by,
            name=name,
            relationship=relationship,
            aliases=[reference] if reference.lower() != name.lower() else None,
            relationship_stated_by=relationship_stated_by,
        )

        return PersonResolutionResult(
            person_id=person.id,
            created=True,
            person_name=person.name,
        )

    async def resolve_names(self, person_ids: list[str]) -> dict[str, str]:
        """Resolve person IDs to names."""
        result: dict[str, str] = {}
        for pid in person_ids:
            person = await self.get(pid)
            if person:
                result[pid] = person.name
        return result

    async def find_person_ids_for_username(
        self,
        username: str,
    ) -> set[str]:
        """Find all person IDs matching a username.

        Searches all records globally. Remaps merged IDs to their primary.

        Args:
            username: Username to match against (case-insensitive).

        Returns:
            Set of matching person IDs (remapped through merge chain).
        """
        username_clean = username.lstrip("@").lower()
        people = await self._ensure_loaded()

        matching: set[str] = set()
        for person in people:
            if self.matches_username(person, username_clean):
                if person.merged_into:
                    # Remap to primary
                    primary = await self._follow_merge_chain(person)
                    matching.add(primary.id)
                else:
                    matching.add(person.id)

        return matching

    # Matching

    @staticmethod
    def matches_username(person: PersonEntry, username: str) -> bool:
        """Check if a person matches a username (case-insensitive)."""
        username_lower = username.lower()
        if person.name.lower() == username_lower:
            return True
        return any(alias.value.lower() == username_lower for alias in person.aliases)

    # Deduplication

    async def find_dedup_candidates(
        self,
        person_ids: list[str],
        *,
        exclude_self: bool = False,
    ) -> list[tuple[str, str]]:
        """Find merge-worthy pairs among person_ids vs all existing people.

        Two-phase: fast heuristic pre-filter then LLM verification.

        Args:
            person_ids: IDs of newly created people to check for duplicates.
            exclude_self: If True, skip candidates where either person has a
                "self" relationship. Use for unattended background dedup to
                avoid false-positive merges into a user's self-person.

        Returns:
            List of (primary_id, secondary_id) tuples to merge.
        """
        if not self._llm or not self._llm_model:
            return []

        people = await self._ensure_loaded()
        active = [p for p in people if not p.merged_into]
        new_people = [p for p in active if p.id in set(person_ids)]

        if not new_people:
            return []

        # Phase 1: Heuristic pre-filter
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
                    a_rels = {r.relationship.lower() for r in new_person.relationships}
                    b_rels = {r.relationship.lower() for r in existing.relationships}
                    if "self" in a_rels or "self" in b_rels:
                        continue
                if self._heuristic_match(new_person, existing):
                    seen.add(pair_key)
                    candidates.append((new_person, existing))

        if not candidates:
            return []

        # Phase 2: LLM verification
        results: list[tuple[str, str]] = []
        for person_a, person_b in candidates:
            if await self._llm_verify_same_person(person_a, person_b):
                primary_id, secondary_id = self._pick_primary(person_a, person_b)
                results.append((primary_id, secondary_id))

        return results

    @staticmethod
    def _heuristic_match(a: PersonEntry, b: PersonEntry) -> bool:
        """Check if two people are potential duplicates via fast heuristics.

        Skips pairs where both have relationship "self".
        """
        # Skip if both are self-persons
        a_rels = {r.relationship.lower() for r in a.relationships}
        b_rels = {r.relationship.lower() for r in b.relationships}
        if "self" in a_rels and "self" in b_rels:
            return False

        a_aliases = {alias.value.lower() for alias in a.aliases}
        b_aliases = {alias.value.lower() for alias in b.aliases}

        # Shared alias (case-insensitive)
        if a_aliases & b_aliases:
            return True

        a_name = a.name.lower()
        b_name = b.name.lower()

        # One's name matches another's relationship
        if a_name in b_rels or b_name in a_rels:
            return True

        # Name substring: one name contains the other
        if len(a_name) >= 3 and len(b_name) >= 3:
            if a_name in b_name or b_name in a_name:
                return True

        # First/last name match: only when one name is a single word.
        # Catches "David" ↔ "David Cramer" but not "David Chen" ↔ "David Cramer"
        # (two multi-word names sharing only a common first name).
        a_parts = set(a_name.split())
        b_parts = set(b_name.split())
        if (len(a_parts) == 1 or len(b_parts) == 1) and a_parts & b_parts:
            return True

        return False

    async def _llm_verify_same_person(self, a: PersonEntry, b: PersonEntry) -> bool:
        """Ask LLM to verify whether two person records are the same person."""
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
            logger.debug("llm_verify_same_person_failed", exc_info=True)
            return False

    @staticmethod
    def _pick_primary(a: PersonEntry, b: PersonEntry) -> tuple[str, str]:
        """Pick which person should be primary in a merge.

        Primary = person with more aliases + relationships.
        Tie-break by older created_at.

        Returns:
            (primary_id, secondary_id)
        """
        _epoch = datetime.min.replace(tzinfo=UTC)

        def _sort_key(p: PersonEntry) -> tuple[int, datetime]:
            score = len(p.aliases) + len(p.relationships)
            # Negate score so higher scores sort first; older dates sort first
            return (-score, p.created_at or _epoch)

        first, second = sorted([a, b], key=_sort_key)
        return first.id, second.id

    # Internal helpers

    @staticmethod
    def _find_by_id(people: list[PersonEntry], person_id: str) -> PersonEntry | None:
        """Find a person by ID in a list."""
        for p in people:
            if p.id == person_id:
                return p
        return None

    async def _follow_merge_chain(self, person: PersonEntry) -> PersonEntry:
        """Follow merge chain to find the primary person."""
        visited: set[str] = set()
        current = person
        while current.merged_into and current.merged_into not in visited:
            visited.add(current.id)
            next_person = await self.get(current.merged_into)
            if not next_person:
                break
            current = next_person
        return current

    async def _fuzzy_find(
        self,
        reference: str,
        content_hint: str | None = None,
        speaker: str | None = None,
    ) -> PersonEntry | None:
        """LLM-assisted fuzzy person matching.

        Called when exact match fails. Asks the LLM to identify which known
        person (if any) the reference refers to.

        Args:
            reference: The person reference to match.
            content_hint: Optional surrounding content for context
                (e.g., "my wife Sarah's birthday is March 15").
            speaker: Optional speaker identifier (username or user ID).
                Included in the prompt so the LLM can prefer relationships
                stated by the current speaker.

        Returns None if no LLM, no candidates, or no match found.
        """
        if not self._llm or not self._llm_model:
            return None

        people = await self._ensure_loaded()
        candidates = [p for p in people if not p.merged_into]
        if not candidates:
            return None

        try:
            from ash.llm.types import Message, Role

            # Build people list for prompt (includes stated_by provenance)
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

            # Validate returned ID against actual people
            return self._find_by_id(candidates, result)
        except Exception:
            logger.debug("fuzzy_find_failed", exc_info=True)
            return None

    @staticmethod
    def _normalize_reference(text: str) -> str:
        """Normalize a person reference by removing common prefixes."""
        result = text.lower().strip()
        for prefix in ["my ", "the ", "@"]:
            if result.startswith(prefix):
                result = result[len(prefix) :]
        return result

    def _parse_person_reference(
        self,
        reference: str,
        content_hint: str | None = None,
    ) -> tuple[str, str | None]:
        """Parse a person reference into name and relationship."""
        ref_lower = reference.lower().strip()

        if ref_lower.startswith("@"):
            ref_lower = ref_lower[1:]

        relationship = ref_lower[3:] if ref_lower.startswith("my ") else None

        if relationship and relationship in RELATIONSHIP_TERMS:
            if content_hint:
                name = self._extract_name_from_content(content_hint, relationship)
                if name:
                    return name, relationship
            return relationship.title(), relationship

        return ref_lower.title(), None

    @staticmethod
    def _extract_name_from_content(
        content: str,
        relationship: str,
    ) -> str | None:
        """Try to extract a person's name from content.

        Supports multi-word names like "Sarah Jane" by capturing consecutive
        capitalized words after a relationship term.
        """
        # Pattern: "wife's name is Sarah Jane" or "wife is named Sarah Jane"
        match = re.search(
            rf"{relationship}(?:'s name is| is named)\s+",
            content,
            re.IGNORECASE,
        )
        if match:
            name = _extract_capitalized_name(content[match.end() :])
            if name:
                return name

        # Pattern: "my wife Sarah Jane loves hiking"
        match = re.search(
            rf"(?:^|,\s*)my {relationship}\s+",
            content,
            re.IGNORECASE,
        )
        if match:
            name = _extract_capitalized_name(content[match.end() :])
            if name:
                return name

        # Pattern: "Sarah's ..." at start of content
        match = re.search(r"^(\w+)'s\s", content)
        if match:
            name = match.group(1)
            if name.lower() not in ["user", "my", "the", "their", "his", "her"]:
                return name

        return None


def create_person_manager(
    people_path: Path | None = None,
) -> PersonManager:
    """Create a PersonManager instance.

    Args:
        people_path: Path to people.jsonl (default: ~/.ash/graph/people.jsonl).

    Returns:
        Configured PersonManager instance.
    """
    return PersonManager(people_path=people_path)
