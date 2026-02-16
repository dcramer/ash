"""Person deduplication and merge operations."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ash.graph.edges import (
    create_merged_into_edge,
    get_merged_into,
)
from ash.store.people.helpers import (
    normalize_reference,
    primary_sort_key,
)
from ash.store.types import AliasEntry, PersonEntry

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
        primary = self._graph.people.get(primary_id)
        secondary = self._graph.people.get(secondary_id)
        if not primary or not secondary:
            return None
        if get_merged_into(self._graph, secondary_id):
            logger.debug(
                "Skipping merge: secondary %s already merged",
                secondary_id,
            )
            return None

        now = datetime.now(UTC)

        # Merge aliases
        existing_values = {a.value.lower() for a in primary.aliases}
        for alias in secondary.aliases:
            if alias.value.lower() not in existing_values:
                primary.aliases.append(
                    AliasEntry(
                        value=alias.value,
                        added_by=alias.added_by,
                        created_at=alias.created_at,
                    )
                )
                existing_values.add(alias.value.lower())

        # Add secondary name as alias if different
        if (
            secondary.name.lower() != primary.name.lower()
            and secondary.name.lower() not in existing_values
        ):
            primary.aliases.append(
                AliasEntry(
                    value=secondary.name,
                    added_by=None,
                    created_at=now,
                )
            )

        # Merge relationships
        existing_rels = {r.relationship.lower() for r in primary.relationships}
        for rc in secondary.relationships:
            if rc.relationship.lower() not in existing_rels:
                primary.relationships.append(rc)
                existing_rels.add(rc.relationship.lower())

        primary.updated_at = now

        # Create MERGED_INTO edge in the knowledge graph
        self._graph.add_edge(create_merged_into_edge(secondary_id, primary_id))
        self._persistence.mark_dirty("people", "edges")

        logger.debug(
            "person_merged",
            extra={"primary_id": primary_id, "secondary_id": secondary_id},
        )

        # Auto-remap memory references and other edges
        try:
            remapped = self._remap_subject_person_id_batched(secondary_id, primary_id)
            if remapped:
                logger.debug(
                    "Remapped %d memories from %s to %s",
                    remapped,
                    secondary_id,
                    primary_id,
                )
        except Exception:
            logger.warning("Failed to remap memories after merge", exc_info=True)

        try:
            edge_remapped = self._remap_edges_for_merge_batched(
                secondary_id, primary_id
            )
            if edge_remapped:
                logger.debug(
                    "Remapped %d edges from %s to %s",
                    edge_remapped,
                    secondary_id,
                    primary_id,
                )
        except Exception:
            logger.warning("Failed to remap edges after merge", exc_info=True)

        # Single flush for all merge mutations
        await self._persistence.flush(self._graph)

        return self._graph.people.get(primary_id)

    async def find_dedup_candidates(
        self: Store,
        person_ids: list[str],
        *,
        exclude_self: bool = False,
    ) -> list[tuple[str, str]]:
        if not self._llm or not self._llm_model:
            return []
        people = await self.get_all_people()
        active = [p for p in people if not get_merged_into(self._graph, p.id)]
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

    async def _remap_edges_for_merge(
        self: Store, secondary_id: str, primary_id: str
    ) -> int:
        """Remap edges and persist immediately."""
        count = self._remap_edges_for_merge_batched(secondary_id, primary_id)
        if count > 0:
            await self._persistence.flush(self._graph)
        return count

    def _remap_edges_for_merge_batched(
        self: Store, secondary_id: str, primary_id: str
    ) -> int:
        """Remap STATED_BY, IS_PERSON, and HAS_RELATIONSHIP edges. Marks dirty; caller must flush."""
        from ash.graph.edges import (
            HAS_RELATIONSHIP,
            IS_PERSON,
            STATED_BY,
            create_has_relationship_edge,
            create_is_person_edge,
            create_stated_by_edge,
        )

        count = 0
        edges_changed = False

        # Remap STATED_BY edges: memory -> secondary becomes memory -> primary
        for edge in list(self._graph.get_incoming(secondary_id, edge_type=STATED_BY)):
            existing = [
                e
                for e in self._graph.get_outgoing(edge.source_id, edge_type=STATED_BY)
                if e.target_id == primary_id
            ]
            self._graph.remove_edge(edge.id)
            if not existing:
                self._graph.add_edge(
                    create_stated_by_edge(
                        edge.source_id, primary_id, created_by="merge"
                    )
                )
            count += 1
            edges_changed = True

        # Remap IS_PERSON edges: user -> secondary becomes user -> primary
        for edge in list(self._graph.get_incoming(secondary_id, edge_type=IS_PERSON)):
            existing = [
                e
                for e in self._graph.get_outgoing(edge.source_id, edge_type=IS_PERSON)
                if e.target_id == primary_id
            ]
            self._graph.remove_edge(edge.id)
            if not existing:
                self._graph.add_edge(create_is_person_edge(edge.source_id, primary_id))
            count += 1
            edges_changed = True

        # Remap HAS_RELATIONSHIP edges: substitute secondary with primary
        for edge in list(
            self._graph.get_outgoing(secondary_id, edge_type=HAS_RELATIONSHIP)
        ):
            other_id = edge.target_id
            existing = [
                e
                for e in self._graph.get_outgoing(
                    primary_id, edge_type=HAS_RELATIONSHIP
                )
                if e.target_id == other_id
            ]
            self._graph.remove_edge(edge.id)
            if not existing and other_id != primary_id:
                props = edge.properties or {}
                self._graph.add_edge(
                    create_has_relationship_edge(
                        primary_id,
                        other_id,
                        relationship_type=props.get("relationship_type"),
                        stated_by=props.get("stated_by"),
                    )
                )
            count += 1
            edges_changed = True

        for edge in list(
            self._graph.get_incoming(secondary_id, edge_type=HAS_RELATIONSHIP)
        ):
            other_id = edge.source_id
            existing = [
                e
                for e in self._graph.get_incoming(
                    primary_id, edge_type=HAS_RELATIONSHIP
                )
                if e.source_id == other_id
            ]
            self._graph.remove_edge(edge.id)
            if not existing and other_id != primary_id:
                props = edge.properties or {}
                self._graph.add_edge(
                    create_has_relationship_edge(
                        other_id,
                        primary_id,
                        relationship_type=props.get("relationship_type"),
                        stated_by=props.get("stated_by"),
                    )
                )
            count += 1
            edges_changed = True

        if edges_changed:
            self._persistence.mark_dirty("edges")

        return count

    async def _follow_merge_chain(self: Store, person: PersonEntry) -> PersonEntry:
        from ash.graph.edges import follow_merge_chain

        canonical_id = follow_merge_chain(self._graph, person.id)
        return self._graph.people.get(canonical_id) or person

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
