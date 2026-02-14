"""Tests for person/identity management subsystem.

Tests focus on:
- PersonManager CRUD operations (global model)
- Alias management with provenance
- Relationship management with provenance
- Username matching and resolution
- Person resolution (resolve_or_create)
- find_person_ids_for_username
- Person merging
"""

from datetime import UTC
from pathlib import Path

import pytest

from ash.core.agent import (
    _build_owner_matchers,
    _extract_relationship_term,
    _is_owner_name,
)
from ash.people import PersonManager
from ash.people.types import AliasEntry, PersonEntry, RelationshipClaim


@pytest.fixture
def person_manager(tmp_path: Path) -> PersonManager:
    """Create a PersonManager with a temporary people.jsonl."""
    return PersonManager(people_path=tmp_path / "people.jsonl")


class TestPersonCRUD:
    """Tests for person CRUD operations."""

    async def test_create_person(self, person_manager: PersonManager):
        """Test creating a person."""
        person = await person_manager.create(
            created_by="user-1",
            name="Sarah",
            relationship="wife",
            aliases=["my wife"],
        )

        assert person.id is not None
        assert person.name == "Sarah"
        assert len(person.relationships) == 1
        assert person.relationships[0].relationship == "wife"
        assert len(person.aliases) == 1
        assert person.aliases[0].value == "my wife"
        assert person.aliases[0].added_by == "user-1"

    async def test_create_person_tracks_created_by(self, person_manager: PersonManager):
        """Test that created_by is set on new records."""
        person = await person_manager.create(
            created_by="dcramer",
            name="Sarah",
        )

        assert person.created_by == "dcramer"

    async def test_get_person(self, person_manager: PersonManager):
        """Test getting a person by ID."""
        created = await person_manager.create(
            created_by="user-1",
            name="Sarah",
        )

        found = await person_manager.get(created.id)

        assert found is not None
        assert found.id == created.id
        assert found.name == "Sarah"

    async def test_find_person_by_name(self, person_manager: PersonManager):
        """Test finding a person by name (globally)."""
        await person_manager.create(
            created_by="user-1",
            name="Sarah",
        )

        found = await person_manager.find("Sarah")

        assert found is not None
        assert found.name == "Sarah"

    async def test_find_person_by_relationship(self, person_manager: PersonManager):
        """Test finding a person by relationship."""
        await person_manager.create(
            created_by="user-1",
            name="Sarah",
            relationship="wife",
        )

        found = await person_manager.find("wife")

        assert found is not None
        assert found.name == "Sarah"

    async def test_find_person_by_alias(self, person_manager: PersonManager):
        """Test finding a person by alias."""
        await person_manager.create(
            created_by="user-1",
            name="Sarah",
            aliases=["my wife"],
        )

        found = await person_manager.find("my wife")

        assert found is not None
        assert found.name == "Sarah"

    async def test_find_person_case_insensitive(self, person_manager: PersonManager):
        """Test that person lookup is case-insensitive."""
        await person_manager.create(
            created_by="user-1",
            name="Sarah",
        )

        found = await person_manager.find("sarah")
        assert found is not None

        found = await person_manager.find("SARAH")
        assert found is not None

    async def test_find_is_global(self, person_manager: PersonManager):
        """Test that find searches all records regardless of creator."""
        await person_manager.create(
            created_by="user-1",
            name="Sarah",
        )

        # Can be found without specifying an owner
        found = await person_manager.find("Sarah")
        assert found is not None

    async def test_list_all(self, person_manager: PersonManager):
        """Test listing all people, sorted by name."""
        await person_manager.create(created_by="user-1", name="Charlie")
        await person_manager.create(created_by="user-1", name="Alice")
        await person_manager.create(created_by="user-2", name="Bob")

        all_people = await person_manager.list_all()

        assert len(all_people) == 3
        assert all_people[0].name == "Alice"
        assert all_people[1].name == "Bob"
        assert all_people[2].name == "Charlie"

    async def test_update_person(self, person_manager: PersonManager):
        """Test updating person details."""
        person = await person_manager.create(
            created_by="user-1",
            name="Sarah",
        )

        updated = await person_manager.update(
            person_id=person.id,
            name="Sara",
            relationship="wife",
        )

        assert updated is not None
        assert updated.name == "Sara"
        assert len(updated.relationships) == 1
        assert updated.relationships[0].relationship == "wife"

    async def test_get_all(self, person_manager: PersonManager):
        """Test getting all people across owners."""
        await person_manager.create(created_by="user-1", name="Alice")
        await person_manager.create(created_by="user-2", name="Bob")

        all_people = await person_manager.get_all()

        assert len(all_people) == 2


class TestAliasManagement:
    """Tests for alias management."""

    async def test_add_alias(self, person_manager: PersonManager):
        """Test adding an alias to a person."""
        person = await person_manager.create(
            created_by="user-1",
            name="Sarah",
        )

        updated = await person_manager.add_alias(person.id, "honey", "user-1")

        assert updated is not None
        assert any(a.value == "honey" for a in updated.aliases)

    async def test_add_alias_no_duplicates(self, person_manager: PersonManager):
        """Test that duplicate aliases are not added."""
        person = await person_manager.create(
            created_by="user-1",
            name="Sarah",
            aliases=["honey"],
        )

        updated = await person_manager.add_alias(person.id, "Honey", "user-1")

        assert updated is not None
        # Should still only have one alias (case-insensitive dedup)
        assert len(updated.aliases) == 1

    async def test_add_alias_anyone_can_add(self, person_manager: PersonManager):
        """Test that any user can add an alias (no owner check)."""
        person = await person_manager.create(
            created_by="user-1",
            name="Sarah",
        )

        result = await person_manager.add_alias(person.id, "honey", "user-2")
        assert result is not None
        assert any(a.value == "honey" for a in result.aliases)


class TestAliasProvenance:
    """Tests for alias provenance tracking."""

    async def test_alias_tracks_added_by(self, person_manager: PersonManager):
        """Test that aliases record who added them."""
        person = await person_manager.create(
            created_by="user-1",
            name="Sarah",
        )

        updated = await person_manager.add_alias(person.id, "sksembhi", "user-2")

        assert updated is not None
        alias = next(a for a in updated.aliases if a.value == "sksembhi")
        assert alias.added_by == "user-2"
        assert alias.created_at is not None

    async def test_create_aliases_track_added_by(self, person_manager: PersonManager):
        """Test that aliases from create() have provenance."""
        person = await person_manager.create(
            created_by="dcramer",
            name="Sarah",
            aliases=["my wife"],
        )

        assert person.aliases[0].value == "my wife"
        assert person.aliases[0].added_by == "dcramer"

    async def test_backward_compat_plain_string_aliases(self, tmp_path: Path):
        """Test that old plain-string aliases load correctly."""
        old_data = {
            "id": "test-123",
            "version": 1,
            "owner_user_id": "user-1",
            "name": "Sarah",
            "aliases": ["my wife", "honey"],
        }

        entry = PersonEntry.from_dict(old_data)

        assert len(entry.aliases) == 2
        assert entry.aliases[0].value == "my wife"
        assert entry.aliases[0].added_by is None
        assert entry.aliases[1].value == "honey"
        # created_by should be migrated from owner_user_id
        assert entry.created_by == "user-1"


class TestRelationshipProvenance:
    """Tests for relationship provenance tracking."""

    async def test_relationship_tracks_stated_by(self, person_manager: PersonManager):
        """Test that relationships record who stated them."""
        person = await person_manager.create(
            created_by="dcramer",
            name="Sarah",
            relationship="wife",
        )

        assert len(person.relationships) == 1
        rc = person.relationships[0]
        assert rc.relationship == "wife"
        assert rc.stated_by == "dcramer"
        assert rc.created_at is not None

    async def test_add_relationship(self, person_manager: PersonManager):
        """Test adding a relationship claim."""
        person = await person_manager.create(
            created_by="user-1",
            name="Sarah",
        )

        updated = await person_manager.add_relationship(person.id, "wife", "dcramer")

        assert updated is not None
        assert len(updated.relationships) == 1
        assert updated.relationships[0].relationship == "wife"
        assert updated.relationships[0].stated_by == "dcramer"

    async def test_add_relationship_no_duplicates(self, person_manager: PersonManager):
        """Test that duplicate relationships are not added."""
        person = await person_manager.create(
            created_by="user-1",
            name="Sarah",
            relationship="wife",
        )

        updated = await person_manager.add_relationship(person.id, "Wife", "user-2")

        assert updated is not None
        assert len(updated.relationships) == 1

    async def test_backward_compat_plain_string_relationship(self):
        """Test that old single-string relationship loads correctly."""
        old_data = {
            "id": "test-123",
            "version": 1,
            "owner_user_id": "user-1",
            "name": "Sarah",
            "relationship": "wife",
        }

        entry = PersonEntry.from_dict(old_data)

        assert len(entry.relationships) == 1
        assert entry.relationships[0].relationship == "wife"
        assert entry.relationships[0].stated_by is None

    async def test_backward_compat_relation_key(self):
        """Test that old 'relation' key also works."""
        old_data = {
            "id": "test-123",
            "version": 1,
            "name": "Sarah",
            "relation": "wife",
        }

        entry = PersonEntry.from_dict(old_data)
        assert len(entry.relationships) == 1
        assert entry.relationships[0].relationship == "wife"


class TestUsernameMatching:
    """Tests for username matching."""

    async def test_matches_username_by_name(self, person_manager: PersonManager):
        """Test matching by name."""
        person = await person_manager.create(
            created_by="user-1",
            name="Bob",
        )

        assert PersonManager.matches_username(person, "bob")
        assert PersonManager.matches_username(person, "Bob")
        assert not PersonManager.matches_username(person, "alice")

    async def test_matches_username_by_alias(self, person_manager: PersonManager):
        """Test matching by alias."""
        person = await person_manager.create(
            created_by="user-1",
            name="Bob",
            aliases=["@bobby", "robert"],
        )

        assert PersonManager.matches_username(person, "@bobby")
        assert PersonManager.matches_username(person, "robert")


class TestResolution:
    """Tests for person resolution."""

    async def test_resolve_existing(self, person_manager: PersonManager):
        """Test resolving an existing person."""
        created = await person_manager.create(
            created_by="user-1",
            name="Sarah",
        )

        result = await person_manager.resolve_or_create("user-1", "Sarah")

        assert result.person_id == created.id
        assert result.created is False
        assert result.person_name == "Sarah"

    async def test_resolve_creates_new(self, person_manager: PersonManager):
        """Test that resolve creates a new person if not found."""
        result = await person_manager.resolve_or_create("user-1", "Bob")

        assert result.created is True
        assert result.person_name == "Bob"

        # Verify it was persisted
        found = await person_manager.get(result.person_id)
        assert found is not None
        assert found.name == "Bob"

    async def test_resolve_relationship_term(self, person_manager: PersonManager):
        """Test resolving a relationship term (e.g., 'my wife')."""
        result = await person_manager.resolve_or_create(
            "user-1",
            "my wife",
            content_hint="My wife Sarah loves hiking",
        )

        assert result.created is True
        assert result.person_name == "Sarah"

        person = await person_manager.get(result.person_id)
        assert person is not None
        assert len(person.relationships) == 1
        assert person.relationships[0].relationship == "wife"

    async def test_resolve_names(self, person_manager: PersonManager):
        """Test resolving person IDs to names."""
        p1 = await person_manager.create(created_by="user-1", name="Alice")
        p2 = await person_manager.create(created_by="user-1", name="Bob")

        names = await person_manager.resolve_names([p1.id, p2.id])

        assert names == {p1.id: "Alice", p2.id: "Bob"}

    async def test_resolve_names_missing_id(self, person_manager: PersonManager):
        """Test that missing IDs are omitted from results."""
        p1 = await person_manager.create(created_by="user-1", name="Alice")

        names = await person_manager.resolve_names([p1.id, "nonexistent"])

        assert names == {p1.id: "Alice"}

    async def test_resolve_global(self, person_manager: PersonManager):
        """Test that resolution finds people created by other users."""
        created = await person_manager.create(
            created_by="user-1",
            name="Sarah",
        )

        # A different user resolving the same name should find the existing person
        result = await person_manager.resolve_or_create("user-2", "Sarah")

        assert result.person_id == created.id
        assert result.created is False


class TestFindPersonIdsForUsername:
    """Tests for find_person_ids_for_username."""

    async def test_finds_globally(self, person_manager: PersonManager):
        """Test finding person IDs across all creators."""
        p1 = await person_manager.create(
            created_by="alice", name="Bob", aliases=["bob"]
        )
        p2 = await person_manager.create(
            created_by="carol", name="Bob2", aliases=["bob"]
        )

        ids = await person_manager.find_person_ids_for_username("bob")

        assert ids == {p1.id, p2.id}

    async def test_handles_at_prefix(self, person_manager: PersonManager):
        """Test that @ prefix is stripped during matching."""
        p = await person_manager.create(created_by="alice", name="Bob", aliases=["bob"])

        ids = await person_manager.find_person_ids_for_username("@bob")

        assert ids == {p.id}

    async def test_returns_empty_for_unknown(self, person_manager: PersonManager):
        """Test that unknown username returns empty set."""
        ids = await person_manager.find_person_ids_for_username("nobody")

        assert ids == set()

    async def test_remaps_merged_ids(self, person_manager: PersonManager):
        """Test that merged person IDs are remapped to primary."""
        p1 = await person_manager.create(
            created_by="user-1", name="Sarah", aliases=["sarah"]
        )
        p2 = await person_manager.create(
            created_by="user-1", name="Sksembhi", aliases=["sksembhi"]
        )

        # Merge p2 into p1
        await person_manager.merge(p1.id, p2.id)

        # Looking up "sksembhi" should return p1's ID (the primary)
        ids = await person_manager.find_person_ids_for_username("sksembhi")
        assert ids == {p1.id}


class TestPersonMerge:
    """Tests for person merging."""

    async def test_merge_combines_aliases(self, person_manager: PersonManager):
        """Test that merge copies aliases from secondary to primary."""
        p1 = await person_manager.create(
            created_by="user-1", name="Sarah", aliases=["my wife"]
        )
        p2 = await person_manager.create(
            created_by="user-2", name="Sksembhi", aliases=["sksembhi"]
        )

        result = await person_manager.merge(p1.id, p2.id)

        assert result is not None
        alias_values = [a.value for a in result.aliases]
        assert "my wife" in alias_values
        assert "sksembhi" in alias_values

    async def test_merge_sets_merged_into(self, person_manager: PersonManager):
        """Test that secondary gets merged_into set."""
        p1 = await person_manager.create(created_by="user-1", name="Sarah")
        p2 = await person_manager.create(created_by="user-2", name="Sksembhi")

        await person_manager.merge(p1.id, p2.id)

        secondary = await person_manager.get(p2.id)
        assert secondary is not None
        assert secondary.merged_into == p1.id

    async def test_merge_adds_secondary_name_as_alias(
        self, person_manager: PersonManager
    ):
        """Test that secondary's name becomes an alias on primary."""
        p1 = await person_manager.create(created_by="user-1", name="Sarah")
        p2 = await person_manager.create(created_by="user-2", name="Sksembhi")

        result = await person_manager.merge(p1.id, p2.id)

        assert result is not None
        alias_values = [a.value for a in result.aliases]
        assert "Sksembhi" in alias_values

    async def test_merge_does_not_duplicate_same_name(
        self, person_manager: PersonManager
    ):
        """Test that merge doesn't add name as alias if same as primary."""
        p1 = await person_manager.create(created_by="user-1", name="Sarah")
        p2 = await person_manager.create(created_by="user-2", name="Sarah")

        result = await person_manager.merge(p1.id, p2.id)

        assert result is not None
        # Should not have "Sarah" as an alias since it matches primary name
        alias_values = [a.value.lower() for a in result.aliases]
        assert "sarah" not in alias_values

    async def test_merged_person_excluded_from_list(
        self, person_manager: PersonManager
    ):
        """Test that merged records don't appear in list_all."""
        p1 = await person_manager.create(created_by="user-1", name="Sarah")
        p2 = await person_manager.create(created_by="user-2", name="Sksembhi")

        await person_manager.merge(p1.id, p2.id)

        people = await person_manager.list_all()
        assert len(people) == 1
        assert people[0].id == p1.id

    async def test_merged_person_excluded_from_find(
        self, person_manager: PersonManager
    ):
        """Test that merged records don't appear in find results."""
        p1 = await person_manager.create(created_by="user-1", name="Sarah")
        p2 = await person_manager.create(
            created_by="user-2", name="Sksembhi", aliases=["sks"]
        )

        await person_manager.merge(p1.id, p2.id)

        # Finding by secondary's name should not return the merged record
        found = await person_manager.find("Sksembhi")
        # It should find p1 via the alias added during merge
        assert found is not None
        assert found.id == p1.id

    async def test_resolve_follows_merge_chain(self, person_manager: PersonManager):
        """Test that resolve_or_create follows merge chain."""
        p1 = await person_manager.create(created_by="user-1", name="Sarah")
        p2 = await person_manager.create(
            created_by="user-2", name="Sksembhi", aliases=["sks"]
        )

        await person_manager.merge(p1.id, p2.id)

        # Resolving by an alias that exists on primary (from merge) should return primary
        result = await person_manager.resolve_or_create("user-3", "Sksembhi")
        assert result.person_id == p1.id
        assert result.created is False

    async def test_merge_copies_relationships(self, person_manager: PersonManager):
        """Test that merge copies relationships from secondary."""
        p1 = await person_manager.create(created_by="user-1", name="Sarah")
        p2 = await person_manager.create(
            created_by="user-2", name="Sksembhi", relationship="wife"
        )

        result = await person_manager.merge(p1.id, p2.id)

        assert result is not None
        assert len(result.relationships) == 1
        assert result.relationships[0].relationship == "wife"

    async def test_merge_returns_none_for_missing(self, person_manager: PersonManager):
        """Test that merge returns None if either ID is missing."""
        p1 = await person_manager.create(created_by="user-1", name="Sarah")

        result = await person_manager.merge(p1.id, "nonexistent")
        assert result is None

    async def test_merge_refuses_already_merged_secondary(
        self, person_manager: PersonManager
    ):
        """Merge should refuse if secondary is already merged (prevents chain corruption)."""
        p1 = await person_manager.create(created_by="user-1", name="Sarah")
        p2 = await person_manager.create(created_by="user-1", name="Sksembhi")
        p3 = await person_manager.create(created_by="user-1", name="Charlie")

        # Merge p2 into p1
        await person_manager.merge(p1.id, p2.id)

        # Now try to merge p2 into p3 — should be refused
        result = await person_manager.merge(p3.id, p2.id)
        assert result is None

        # p2 should still point to p1, not p3
        p2_after = await person_manager.get(p2.id)
        assert p2_after is not None
        assert p2_after.merged_into == p1.id


class TestCacheBehavior:
    """Tests for cache invalidation."""

    async def test_cache_invalidates_after_create(self, person_manager: PersonManager):
        """Test that cache is invalidated after creating a person."""
        await person_manager.create(created_by="user-1", name="Alice")
        people1 = await person_manager.get_all()
        assert len(people1) == 1

        await person_manager.create(created_by="user-1", name="Bob")
        people2 = await person_manager.get_all()
        assert len(people2) == 2

    async def test_cache_invalidates_after_update(self, person_manager: PersonManager):
        """Test that cache is invalidated after update."""
        person = await person_manager.create(created_by="user-1", name="Alice")

        await person_manager.update(person_id=person.id, name="Alicia")

        refreshed = await person_manager.get(person.id)
        assert refreshed is not None
        assert refreshed.name == "Alicia"


class TestSerialization:
    """Tests for to_dict/from_dict roundtripping."""

    def test_roundtrip(self):
        """Test that PersonEntry survives serialization."""
        entry = PersonEntry(
            id="test-123",
            version=1,
            created_by="dcramer",
            name="Sarah",
            relationships=[
                RelationshipClaim(relationship="wife", stated_by="dcramer"),
            ],
            aliases=[
                AliasEntry(value="my wife", added_by="dcramer"),
                AliasEntry(value="sksembhi", added_by="sksembhi"),
            ],
        )

        d = entry.to_dict()
        restored = PersonEntry.from_dict(d)

        assert restored.id == entry.id
        assert restored.created_by == entry.created_by
        assert restored.name == entry.name
        assert len(restored.relationships) == 1
        assert restored.relationships[0].relationship == "wife"
        assert restored.relationships[0].stated_by == "dcramer"
        assert len(restored.aliases) == 2
        assert restored.aliases[0].value == "my wife"
        assert restored.aliases[1].value == "sksembhi"


class TestNormalizeReference:
    """Tests for _normalize_reference prefix stripping."""

    def test_strips_my_prefix(self):
        assert PersonManager._normalize_reference("my wife") == "wife"

    def test_strips_the_prefix(self):
        assert PersonManager._normalize_reference("the boss") == "boss"

    def test_strips_at_prefix(self):
        assert PersonManager._normalize_reference("@dcramer") == "dcramer"

    def test_strips_my_then_at(self):
        """After stripping 'my ', '@dcramer' should also strip '@'."""
        assert PersonManager._normalize_reference("my @dcramer") == "dcramer"

    def test_no_prefix(self):
        assert PersonManager._normalize_reference("sarah") == "sarah"


class TestParsePersonReference:
    """Tests for _parse_person_reference relationship filtering."""

    def test_valid_relationship_term(self):
        pm = PersonManager()
        name, rel = pm._parse_person_reference("my wife")
        assert rel == "wife"

    def test_invalid_relationship_term_returns_none(self):
        """Non-relationship terms like 'cat' should not be stored as relationships."""
        pm = PersonManager()
        name, rel = pm._parse_person_reference("my cat")
        assert rel is None
        # "my cat" is not a recognized relationship, so full string is title-cased
        assert name == "My Cat"

    def test_plain_name_no_relationship(self):
        pm = PersonManager()
        name, rel = pm._parse_person_reference("Sarah")
        assert rel is None
        assert name == "Sarah"


class TestUpdateProvenance:
    """Tests for update() provenance tracking."""

    async def test_update_relationship_with_provenance(
        self, person_manager: PersonManager
    ):
        person = await person_manager.create(created_by="user-1", name="Sarah")

        updated = await person_manager.update(
            person_id=person.id,
            relationship="wife",
            updated_by="dcramer",
        )

        assert updated is not None
        assert len(updated.relationships) == 1
        rc = updated.relationships[0]
        assert rc.relationship == "wife"
        assert rc.stated_by == "dcramer"
        assert rc.created_at is not None

    async def test_update_aliases_with_provenance(self, person_manager: PersonManager):
        person = await person_manager.create(created_by="user-1", name="Sarah")

        updated = await person_manager.update(
            person_id=person.id,
            aliases=["sksembhi", "honey"],
            updated_by="dcramer",
        )

        assert updated is not None
        assert len(updated.aliases) == 2
        for alias in updated.aliases:
            assert alias.added_by == "dcramer"
            assert alias.created_at is not None

    async def test_update_without_updated_by(self, person_manager: PersonManager):
        """Provenance fields are None when updated_by is not provided."""
        person = await person_manager.create(created_by="user-1", name="Sarah")

        updated = await person_manager.update(
            person_id=person.id,
            relationship="wife",
        )

        assert updated is not None
        assert updated.relationships[0].stated_by is None


class TestPersonDelete:
    """Tests for person deletion."""

    async def test_delete_person(self, person_manager: PersonManager):
        """Test deleting a person."""
        person = await person_manager.create(created_by="user-1", name="Sarah")

        result = await person_manager.delete(person.id)
        assert result is True

        found = await person_manager.get(person.id)
        assert found is None

    async def test_delete_nonexistent(self, person_manager: PersonManager):
        """Test deleting a person that doesn't exist."""
        result = await person_manager.delete("nonexistent")
        assert result is False

    async def test_delete_clears_merged_into(self, person_manager: PersonManager):
        """Test that deleting a person clears merged_into references."""
        p1 = await person_manager.create(created_by="user-1", name="Sarah")
        p2 = await person_manager.create(created_by="user-1", name="Sksembhi")

        await person_manager.merge(p1.id, p2.id)

        # Delete the primary — secondary's merged_into should be cleared
        await person_manager.delete(p1.id)

        p2_after = await person_manager.get(p2.id)
        assert p2_after is not None
        assert p2_after.merged_into is None

    async def test_delete_removes_from_list(self, person_manager: PersonManager):
        """Test that deleted person is gone from list_all."""
        p1 = await person_manager.create(created_by="user-1", name="Alice")
        p2 = await person_manager.create(created_by="user-1", name="Bob")

        await person_manager.delete(p1.id)

        people = await person_manager.list_all()
        assert len(people) == 1
        assert people[0].id == p2.id


class TestMultiWordNameExtraction:
    """Tests for multi-word name extraction from content."""

    def test_extracts_two_word_name(self):
        """Test extracting 'Sarah Jane' from 'my wife Sarah Jane loves hiking'."""
        result = PersonManager._extract_name_from_content(
            "My wife Sarah Jane loves hiking", "wife"
        )
        assert result == "Sarah Jane"

    def test_extracts_single_word_name(self):
        """Test extracting single-word name still works."""
        result = PersonManager._extract_name_from_content(
            "My wife Sarah loves hiking", "wife"
        )
        assert result == "Sarah"

    def test_extracts_name_from_is_named(self):
        """Test 'wife is named Sarah Jane' pattern."""
        result = PersonManager._extract_name_from_content(
            "My wife is named Sarah Jane", "wife"
        )
        assert result == "Sarah Jane"

    def test_extracts_name_from_possessive(self):
        """Test 'wife's name is Sarah Jane' pattern."""
        result = PersonManager._extract_name_from_content(
            "My wife's name is Sarah Jane", "wife"
        )
        assert result == "Sarah Jane"


class TestFuzzyResolution:
    """Tests for fuzzy resolution graceful degradation (no LLM mocking).

    Actual fuzzy matching quality is tested by the identity eval
    (evals/test_identity.py) which uses real LLM calls.
    """

    async def test_no_llm_degrades_to_create(self, person_manager: PersonManager):
        """Without LLM configured, fuzzy match is skipped and a new person is created."""
        await person_manager.create(
            created_by="user-1",
            name="Sukhpreet Sembhi",
        )

        # No set_llm() call — LLM is None
        result = await person_manager.resolve_or_create("user-1", "Sukhpreet")

        assert result.created is True
        assert result.person_name == "Sukhpreet"

    async def test_llm_error_degrades_to_create(self, person_manager: PersonManager):
        """LLM exception degrades gracefully to creating a new person."""
        from unittest.mock import AsyncMock

        await person_manager.create(
            created_by="user-1",
            name="Sukhpreet Sembhi",
        )

        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = RuntimeError("API error")
        person_manager.set_llm(mock_llm, "test-model")

        result = await person_manager.resolve_or_create("user-1", "Sukhpreet")

        assert result.created is True


class TestRelationshipStatedBy:
    """Tests for relationship_stated_by override (Bug 1 fix)."""

    async def test_create_with_relationship_stated_by(
        self, person_manager: PersonManager
    ):
        """stated_by uses the override, not created_by."""
        person = await person_manager.create(
            created_by="123456789",
            name="Sarah",
            relationship="wife",
            relationship_stated_by="dcramer",
        )

        assert person.relationships[0].stated_by == "dcramer"
        assert person.created_by == "123456789"

    async def test_create_relationship_stated_by_fallback(
        self, person_manager: PersonManager
    ):
        """Falls back to created_by when relationship_stated_by not provided."""
        person = await person_manager.create(
            created_by="dcramer",
            name="Sarah",
            relationship="wife",
        )

        assert person.relationships[0].stated_by == "dcramer"

    async def test_resolve_or_create_passes_relationship_stated_by(
        self, person_manager: PersonManager
    ):
        """relationship_stated_by flows through to created person."""
        result = await person_manager.resolve_or_create(
            created_by="123456789",
            reference="my wife",
            content_hint="My wife Sarah loves hiking",
            relationship_stated_by="dcramer",
        )

        assert result.created is True
        person = await person_manager.get(result.person_id)
        assert person is not None
        assert person.relationships[0].stated_by == "dcramer"


class TestOwnerNameMatchers:
    """Tests for _build_owner_matchers name matching (Bug 3 fix).

    Prefers false positives (dropping a valid subject) over false negatives
    (creating a duplicate person entry for the speaker).
    """

    def test_first_name_of_multi_word_matches(self):
        """First name of multi-word name matches as owner."""
        matchers = _build_owner_matchers(["David Cramer", "dcramer"])
        assert _is_owner_name("David", matchers) is True

    def test_last_name_of_multi_word_matches(self):
        """Last name of multi-word name matches as owner."""
        matchers = _build_owner_matchers(["David Cramer"])
        assert _is_owner_name("Cramer", matchers) is True

    def test_single_word_name_no_parts(self):
        """Single-word name does NOT add parts — exact-only matching."""
        matchers = _build_owner_matchers(["dcramer"])
        # "dcramer" is exact match
        assert _is_owner_name("dcramer", matchers) is True
        # No parts were added for single-word names
        assert len(matchers.parts) == 0

    def test_prefix_of_name_part_matches(self):
        """Prefix of a name part matches (e.g., Davi → David)."""
        matchers = _build_owner_matchers(["David Cramer"])
        # "davi" is a prefix of part "david"
        assert _is_owner_name("Davi", matchers) is True
        # "dave" is NOT a prefix of "david" — true nicknames aren't caught
        assert _is_owner_name("Dave", matchers) is False

    def test_subject_starting_with_name_part_matches(self):
        """Subject starting with a name part matches (e.g., 'David C.' → 'david')."""
        matchers = _build_owner_matchers(["David Cramer"])
        assert _is_owner_name("David C.", matchers) is True

    def test_short_subjects_skip_prefix_check(self):
        """Very short subjects (< 3 chars) don't trigger prefix matching."""
        matchers = _build_owner_matchers(["David Cramer"])
        assert _is_owner_name("Da", matchers) is False

    def test_username_not_split(self):
        """Username is not split into parts."""
        matchers = _build_owner_matchers(["dcramer"])
        assert _is_owner_name("dcr", matchers) is False

    def test_full_name_exact_match(self):
        """Full name still matches exactly."""
        matchers = _build_owner_matchers(["David Cramer"])
        assert _is_owner_name("David Cramer", matchers) is True

    def test_at_prefix_stripped(self):
        """@ prefix is stripped before matching."""
        matchers = _build_owner_matchers(["dcramer"])
        assert _is_owner_name("@dcramer", matchers) is True

    def test_unrelated_name_does_not_match(self):
        """Completely unrelated names don't match."""
        matchers = _build_owner_matchers(["David Cramer", "dcramer"])
        assert _is_owner_name("Sarah", matchers) is False
        assert _is_owner_name("Bob Smith", matchers) is False

    def test_short_name_parts_excluded(self):
        """Name parts shorter than 3 chars are excluded from part matching."""
        matchers = _build_owner_matchers(["Li Wei"])
        # "li" is too short (< 3) to be a part
        assert "li" not in matchers.parts
        assert "wei" in matchers.parts


class TestContentAwareFuzzyFind:
    """Tests for content_hint and speaker passing in fuzzy matching."""

    async def test_fuzzy_find_passes_content_hint(self, person_manager: PersonManager):
        """Verify content_hint is forwarded to _fuzzy_find via resolve_or_create."""
        from unittest.mock import AsyncMock, MagicMock

        await person_manager.create(
            created_by="user-1",
            name="Wife",
            relationship="wife",
        )

        # Mock LLM to return matching person ID
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.get_text.return_value = "NONE"
        mock_llm.complete.return_value = mock_response
        person_manager.set_llm(mock_llm, "test-model")

        await person_manager.resolve_or_create(
            "user-1", "Sarah", content_hint="my wife Sarah's birthday is March 15"
        )

        # Verify LLM was called with a prompt that includes the content hint
        call_args = mock_llm.complete.call_args
        prompt = call_args.kwargs["messages"][0].content
        assert "my wife Sarah's birthday is March 15" in prompt

    async def test_fuzzy_find_without_content_hint(self, person_manager: PersonManager):
        """Verify _fuzzy_find works without content_hint (no Context line)."""
        from unittest.mock import AsyncMock, MagicMock

        await person_manager.create(
            created_by="user-1",
            name="Sarah",
        )

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.get_text.return_value = "NONE"
        mock_llm.complete.return_value = mock_response
        person_manager.set_llm(mock_llm, "test-model")

        await person_manager.resolve_or_create("user-1", "Bob")

        call_args = mock_llm.complete.call_args
        prompt = call_args.kwargs["messages"][0].content
        assert "Context:" not in prompt

    async def test_fuzzy_find_includes_speaker(self, person_manager: PersonManager):
        """Speaker is included in the fuzzy match prompt."""
        from unittest.mock import AsyncMock, MagicMock

        await person_manager.create(
            created_by="user-1",
            name="Wife",
            relationship="wife",
            relationship_stated_by="dcramer",
        )

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.get_text.return_value = "NONE"
        mock_llm.complete.return_value = mock_response
        person_manager.set_llm(mock_llm, "test-model")

        await person_manager.resolve_or_create(
            "user-1",
            "Sarah",
            content_hint="Sarah's birthday is March 15",
            relationship_stated_by="dcramer",
        )

        call_args = mock_llm.complete.call_args
        prompt = call_args.kwargs["messages"][0].content
        assert 'Speaker: "dcramer"' in prompt

    async def test_fuzzy_find_includes_stated_by_on_relationships(
        self, person_manager: PersonManager
    ):
        """Relationship stated_by provenance appears in the fuzzy match prompt."""
        from unittest.mock import AsyncMock, MagicMock

        await person_manager.create(
            created_by="user-1",
            name="Wife",
            relationship="wife",
            relationship_stated_by="dcramer",
        )

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.get_text.return_value = "NONE"
        mock_llm.complete.return_value = mock_response
        person_manager.set_llm(mock_llm, "test-model")

        await person_manager.resolve_or_create(
            "user-1", "Sarah", relationship_stated_by="dcramer"
        )

        call_args = mock_llm.complete.call_args
        prompt = call_args.kwargs["messages"][0].content
        assert "stated by dcramer" in prompt

    async def test_fuzzy_find_speaker_falls_back_to_created_by(
        self, person_manager: PersonManager
    ):
        """Without relationship_stated_by, speaker falls back to created_by."""
        from unittest.mock import AsyncMock, MagicMock

        await person_manager.create(created_by="user-1", name="Sarah")

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.get_text.return_value = "NONE"
        mock_llm.complete.return_value = mock_response
        person_manager.set_llm(mock_llm, "test-model")

        await person_manager.resolve_or_create("user-42", "Bob")

        call_args = mock_llm.complete.call_args
        prompt = call_args.kwargs["messages"][0].content
        assert 'Speaker: "user-42"' in prompt


class TestHeuristicMatch:
    """Tests for _heuristic_match dedup heuristics."""

    def test_shared_alias(self):
        """Shared alias should match."""
        a = PersonEntry(
            id="a",
            name="Sarah",
            aliases=[AliasEntry(value="sks")],
        )
        b = PersonEntry(
            id="b",
            name="Sukhpreet",
            aliases=[AliasEntry(value="sks")],
        )
        assert PersonManager._heuristic_match(a, b) is True

    def test_name_matches_relationship(self):
        """One's name matching another's relationship should match."""
        a = PersonEntry(
            id="a",
            name="Wife",
        )
        b = PersonEntry(
            id="b",
            name="Sarah",
            relationships=[RelationshipClaim(relationship="wife")],
        )
        assert PersonManager._heuristic_match(a, b) is True

    def test_name_substring(self):
        """Name substring should match."""
        a = PersonEntry(id="a", name="Sukhpreet")
        b = PersonEntry(id="b", name="Sukhpreet Sembhi")
        assert PersonManager._heuristic_match(a, b) is True

    def test_first_name_overlap(self):
        """First name matching another's full name should match."""
        a = PersonEntry(id="a", name="Sarah")
        b = PersonEntry(id="b", name="Sarah Jane")
        assert PersonManager._heuristic_match(a, b) is True

    def test_skip_both_self(self):
        """Should skip pairs where both have relationship self."""
        a = PersonEntry(
            id="a",
            name="David",
            relationships=[RelationshipClaim(relationship="self")],
        )
        b = PersonEntry(
            id="b",
            name="David Cramer",
            relationships=[RelationshipClaim(relationship="self")],
        )
        assert PersonManager._heuristic_match(a, b) is False

    def test_no_match_unrelated(self):
        """Unrelated people should not match."""
        a = PersonEntry(id="a", name="Sarah")
        b = PersonEntry(id="b", name="Bob")
        assert PersonManager._heuristic_match(a, b) is False

    def test_short_names_no_substring(self):
        """Names shorter than 3 chars should not trigger substring match."""
        a = PersonEntry(id="a", name="Al")
        b = PersonEntry(id="b", name="Alice")
        assert PersonManager._heuristic_match(a, b) is False

    def test_two_multiword_names_shared_first_name_no_match(self):
        """Two multi-word names sharing only a first name should NOT match."""
        a = PersonEntry(id="a", name="David Chen")
        b = PersonEntry(id="b", name="David Cramer")
        assert PersonManager._heuristic_match(a, b) is False

    def test_single_word_vs_multiword_shared_name_matches(self):
        """Single-word name matching a part of multi-word name SHOULD match."""
        a = PersonEntry(id="a", name="David")
        b = PersonEntry(id="b", name="David Cramer")
        assert PersonManager._heuristic_match(a, b) is True


class TestPickPrimary:
    """Tests for _pick_primary merge direction."""

    def test_more_aliases_is_primary(self):
        """Person with more aliases/relationships becomes primary."""
        a = PersonEntry(
            id="a",
            name="Sarah",
            aliases=[AliasEntry(value="sks"), AliasEntry(value="honey")],
            relationships=[RelationshipClaim(relationship="wife")],
        )
        b = PersonEntry(id="b", name="Sukhpreet")
        primary_id, secondary_id = PersonManager._pick_primary(a, b)
        assert primary_id == "a"
        assert secondary_id == "b"

    def test_tie_break_by_created_at(self):
        """On tie, older record is primary."""
        from datetime import datetime

        a = PersonEntry(
            id="a",
            name="Sarah",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        b = PersonEntry(
            id="b",
            name="Sarah Jane",
            created_at=datetime(2024, 6, 1, tzinfo=UTC),
        )
        primary_id, secondary_id = PersonManager._pick_primary(a, b)
        assert primary_id == "a"
        assert secondary_id == "b"


class TestFindDedupCandidates:
    """Tests for find_dedup_candidates."""

    async def test_no_llm_returns_empty(self, person_manager: PersonManager):
        """Without LLM, dedup returns empty list."""
        await person_manager.create(created_by="user-1", name="Sarah")
        p2 = await person_manager.create(created_by="user-1", name="Sarah Jane")

        result = await person_manager.find_dedup_candidates([p2.id])
        assert result == []

    async def test_finds_heuristic_match_with_llm_yes(
        self, person_manager: PersonManager
    ):
        """Heuristic match confirmed by LLM returns merge candidate."""
        from unittest.mock import AsyncMock, MagicMock

        p1 = await person_manager.create(
            created_by="user-1",
            name="Sarah",
            relationship="wife",
        )
        p2 = await person_manager.create(created_by="user-1", name="Sarah Jane")

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.get_text.return_value = "YES"
        mock_llm.complete.return_value = mock_response
        person_manager.set_llm(mock_llm, "test-model")

        result = await person_manager.find_dedup_candidates([p2.id])
        assert len(result) == 1
        # p1 has more data (relationship), so it's primary
        assert result[0] == (p1.id, p2.id)

    async def test_heuristic_match_with_llm_no(self, person_manager: PersonManager):
        """Heuristic match rejected by LLM returns no candidates."""
        from unittest.mock import AsyncMock, MagicMock

        await person_manager.create(created_by="user-1", name="Sarah")
        p2 = await person_manager.create(created_by="user-1", name="Sarah Connor")

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.get_text.return_value = "NO"
        mock_llm.complete.return_value = mock_response
        person_manager.set_llm(mock_llm, "test-model")

        result = await person_manager.find_dedup_candidates([p2.id])
        assert result == []

    async def test_no_heuristic_match_no_llm_call(self, person_manager: PersonManager):
        """No heuristic match means no LLM verification call."""
        from unittest.mock import AsyncMock

        await person_manager.create(created_by="user-1", name="Sarah")
        p2 = await person_manager.create(created_by="user-1", name="Bob")

        mock_llm = AsyncMock()
        person_manager.set_llm(mock_llm, "test-model")

        result = await person_manager.find_dedup_candidates([p2.id])
        assert result == []
        mock_llm.complete.assert_not_called()

    async def test_all_ids_no_duplicate_pairs(self, person_manager: PersonManager):
        """When all IDs are passed (doctor mode), each pair is checked only once."""
        from unittest.mock import AsyncMock, MagicMock

        p1 = await person_manager.create(created_by="user-1", name="Sarah")
        p2 = await person_manager.create(created_by="user-1", name="Sarah Jane")

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.get_text.return_value = "YES"
        mock_llm.complete.return_value = mock_response
        person_manager.set_llm(mock_llm, "test-model")

        # Pass BOTH IDs (doctor mode) — should NOT generate duplicate pairs
        result = await person_manager.find_dedup_candidates([p1.id, p2.id])
        assert len(result) == 1
        # LLM should be called exactly once, not twice
        assert mock_llm.complete.call_count == 1

    async def test_exclude_self_skips_self_person(self, person_manager: PersonManager):
        """exclude_self=True skips candidates where either person has 'self' relationship."""
        from unittest.mock import AsyncMock

        await person_manager.create(
            created_by="user-1",
            name="David Cramer",
            relationship="self",
        )
        p2 = await person_manager.create(created_by="user-2", name="David")

        mock_llm = AsyncMock()
        person_manager.set_llm(mock_llm, "test-model")

        # With exclude_self=True, should skip the candidate
        result = await person_manager.find_dedup_candidates([p2.id], exclude_self=True)
        assert result == []
        mock_llm.complete.assert_not_called()

    async def test_exclude_self_false_allows_self_person(
        self, person_manager: PersonManager
    ):
        """exclude_self=False (default) allows matching against self-persons."""
        from unittest.mock import AsyncMock, MagicMock

        await person_manager.create(
            created_by="user-1",
            name="David Cramer",
            relationship="self",
        )
        p2 = await person_manager.create(created_by="user-2", name="David")

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.get_text.return_value = "YES"
        mock_llm.complete.return_value = mock_response
        person_manager.set_llm(mock_llm, "test-model")

        # Without exclude_self (default=False), should find candidates
        result = await person_manager.find_dedup_candidates([p2.id])
        assert len(result) == 1


class TestAutoRemapOnMerge:
    """Tests for automatic memory remap on merge."""

    async def test_merge_calls_remap(self, person_manager: PersonManager):
        """Merge should call remap_subject_person_id when memory manager is set."""
        from unittest.mock import AsyncMock

        p1 = await person_manager.create(created_by="user-1", name="Sarah")
        p2 = await person_manager.create(created_by="user-1", name="Sksembhi")

        mock_mm = AsyncMock()
        mock_mm.remap_subject_person_id.return_value = 3
        person_manager.set_memory_manager(mock_mm)

        await person_manager.merge(p1.id, p2.id)

        mock_mm.remap_subject_person_id.assert_called_once_with(p2.id, p1.id)

    async def test_merge_without_memory_manager(self, person_manager: PersonManager):
        """Merge should work fine without memory manager."""
        p1 = await person_manager.create(created_by="user-1", name="Sarah")
        p2 = await person_manager.create(created_by="user-1", name="Sksembhi")

        result = await person_manager.merge(p1.id, p2.id)
        assert result is not None
        assert result.id == p1.id

    async def test_merge_remap_error_non_fatal(self, person_manager: PersonManager):
        """Remap failure should not break merge."""
        from unittest.mock import AsyncMock

        p1 = await person_manager.create(created_by="user-1", name="Sarah")
        p2 = await person_manager.create(created_by="user-1", name="Sksembhi")

        mock_mm = AsyncMock()
        mock_mm.remap_subject_person_id.side_effect = RuntimeError("DB error")
        person_manager.set_memory_manager(mock_mm)

        result = await person_manager.merge(p1.id, p2.id)
        # Merge still succeeds even though remap failed
        assert result is not None
        assert result.id == p1.id


class TestSpeakerScopedResolution:
    """Tests for speaker-scoped resolution via find_for_speaker."""

    async def test_alice_and_bob_separate_sarahs(self, person_manager: PersonManager):
        """Alice's wife Sarah and Bob's coworker Sarah are distinct persons."""
        # Alice creates "my wife Sarah"
        alice_result = await person_manager.resolve_or_create(
            created_by="alice-id",
            reference="my wife",
            content_hint="My wife Sarah likes hiking",
            relationship_stated_by="alice-id",
        )

        # Bob creates "my coworker Sarah"
        bob_result = await person_manager.resolve_or_create(
            created_by="bob-id",
            reference="my coworker",
            content_hint="My coworker Sarah is great with databases",
            relationship_stated_by="bob-id",
        )

        # Should be two distinct persons
        assert alice_result.person_id != bob_result.person_id

    async def test_find_for_speaker_returns_connected_person(
        self, person_manager: PersonManager
    ):
        """find_for_speaker returns person connected to speaker via KNOWS edge."""
        # Create Sarah with Alice's relationship
        sarah = await person_manager.create(
            created_by="alice-id",
            name="Sarah",
            relationship="wife",
            relationship_stated_by="alice-id",
        )

        result = await person_manager.find_for_speaker("Sarah", "alice-id")
        assert result is not None
        assert result.id == sarah.id

    async def test_find_for_speaker_by_relationship_term(
        self, person_manager: PersonManager
    ):
        """find_for_speaker matches by relationship term (e.g., 'wife')."""
        sarah = await person_manager.create(
            created_by="alice-id",
            name="Sarah",
            relationship="wife",
            relationship_stated_by="alice-id",
        )

        result = await person_manager.find_for_speaker("wife", "alice-id")
        assert result is not None
        assert result.id == sarah.id

    async def test_find_for_speaker_different_speaker_no_match(
        self, person_manager: PersonManager
    ):
        """find_for_speaker returns None when speaker has no edges to any Sarah."""
        await person_manager.create(
            created_by="alice-id",
            name="Sarah",
            relationship="wife",
            relationship_stated_by="alice-id",
        )

        # Charlie has no edges to any Sarah
        result = await person_manager.find_for_speaker("Sarah", "charlie-id")
        assert result is None

    async def test_find_for_speaker_falls_through_to_global(
        self, person_manager: PersonManager
    ):
        """When speaker has no match, resolve_or_create falls through to global find."""
        sarah = await person_manager.create(
            created_by="alice-id",
            name="Sarah",
        )

        # Charlie resolving "Sarah" with no speaker edges falls to global find
        result = await person_manager.resolve_or_create(
            created_by="charlie-id",
            reference="Sarah",
            relationship_stated_by="charlie-id",
        )

        # Should find the globally existing Sarah
        assert result.person_id == sarah.id
        assert result.created is False

    async def test_find_for_speaker_scopes_by_alias_added_by(
        self, person_manager: PersonManager
    ):
        """find_for_speaker matches via ALIAS edge (added_by)."""
        sarah = await person_manager.create(
            created_by="alice-id",
            name="Sarah",
            aliases=["sks"],
        )

        result = await person_manager.find_for_speaker("sks", "alice-id")
        assert result is not None
        assert result.id == sarah.id

    async def test_resolve_or_create_prefers_speaker_match(
        self, person_manager: PersonManager
    ):
        """resolve_or_create prefers speaker-scoped match over global."""
        # Create two Sarahs - one connected to Alice, one to Bob
        alice_sarah = await person_manager.create(
            created_by="alice-id",
            name="Sarah",
            relationship="wife",
            relationship_stated_by="alice-id",
            aliases=["alice-sarah"],
        )
        await person_manager.create(
            created_by="bob-id",
            name="Sarah B",
            relationship="coworker",
            relationship_stated_by="bob-id",
            aliases=["Sarah"],
        )

        # Alice resolving "Sarah" should find her Sarah (via KNOWS edge)
        result = await person_manager.resolve_or_create(
            created_by="alice-id",
            reference="Sarah",
            relationship_stated_by="alice-id",
        )
        assert result.person_id == alice_sarah.id


class TestNoUsernameFallback:
    """Tests for no-username user fallback (numeric ID as alias)."""

    async def test_self_person_with_no_username_gets_user_id_alias(
        self, person_manager: PersonManager
    ):
        """Self-person created with no username uses numeric user_id as alias."""
        from ash.core.agent import Agent

        # Create a minimal agent with person_manager
        agent = Agent.__new__(Agent)
        agent._people = person_manager

        await agent._ensure_self_person(
            user_id="123456789",
            username="",
            display_name="David Cramer",
        )

        # Should have created a person with user_id as alias
        people = await person_manager.list_all()
        assert len(people) == 1
        assert people[0].name == "David Cramer"
        alias_values = [a.value for a in people[0].aliases]
        assert "123456789" in alias_values

    async def test_no_username_user_id_resolves(self, person_manager: PersonManager):
        """find_person_ids_for_username with numeric ID resolves after no-username create."""
        from ash.core.agent import Agent

        agent = Agent.__new__(Agent)
        agent._people = person_manager

        await agent._ensure_self_person(
            user_id="123456789",
            username="",
            display_name="David Cramer",
        )

        ids = await person_manager.find_person_ids_for_username("123456789")
        assert len(ids) == 1

    async def test_self_person_with_username_uses_username_alias(
        self, person_manager: PersonManager
    ):
        """Self-person created with username uses username as alias (not user_id)."""
        from ash.core.agent import Agent

        agent = Agent.__new__(Agent)
        agent._people = person_manager

        await agent._ensure_self_person(
            user_id="123456789",
            username="notzeeg",
            display_name="David Cramer",
        )

        people = await person_manager.list_all()
        assert len(people) == 1
        alias_values = [a.value for a in people[0].aliases]
        assert "notzeeg" in alias_values
        # user_id should NOT be an alias when username is present
        assert "123456789" not in alias_values


class TestExtractRelationshipTerm:
    """Tests for _extract_relationship_term helper."""

    def test_extracts_wife(self):
        assert _extract_relationship_term("Sarah is the user's wife") == "wife"

    def test_extracts_boss(self):
        assert _extract_relationship_term("John is the boss") == "boss"

    def test_extracts_best_friend_over_friend(self):
        """Multi-word terms matched before single-word substrings."""
        assert _extract_relationship_term("Alex is my best friend") == "best friend"

    def test_no_match(self):
        assert _extract_relationship_term("Sarah likes hiking") is None

    def test_case_insensitive(self):
        assert _extract_relationship_term("Sarah is my Wife") == "wife"
