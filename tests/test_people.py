"""Tests for person/identity management via Store.

Tests focus on:
- Store person CRUD operations (global model)
- Alias management with provenance
- Relationship management with provenance
- Username matching and resolution
- Person resolution (resolve_or_create_person)
- find_person_ids_for_username
- Person merging
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.core.filters import build_owner_matchers, is_owner_name
from ash.db.engine import Database
from ash.memory.embeddings import EmbeddingGenerator
from ash.memory.index import VectorIndex
from ash.memory.processing import extract_relationship_term
from ash.store.store import Store
from ash.store.types import AliasEntry, PersonEntry, RelationshipClaim


def _make_mock_llm(response_text: str = "YES") -> AsyncMock:
    """Create a mock LLM that returns the given text for complete() calls."""
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.message.get_text.return_value = response_text
    mock_llm.complete.return_value = mock_response
    return mock_llm


@pytest.fixture
def mock_index():
    index = MagicMock(spec=VectorIndex)
    index.search = AsyncMock(return_value=[])
    index.add_embedding = AsyncMock()
    index.delete_embedding = AsyncMock()
    index.delete_embeddings = AsyncMock()
    return index


@pytest.fixture
def mock_embedding_generator():
    generator = MagicMock(spec=EmbeddingGenerator)
    generator.embed = AsyncMock(return_value=[0.1] * 1536)
    return generator


@pytest.fixture
async def graph_store(
    database: Database, mock_index, mock_embedding_generator
) -> Store:
    return Store(
        db=database,
        vector_index=mock_index,
        embedding_generator=mock_embedding_generator,
    )


class TestPersonCRUD:
    """Tests for person CRUD operations."""

    async def test_create_person(self, graph_store: Store):
        """Test creating a person."""
        person = await graph_store.create_person(
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

    async def test_create_person_tracks_created_by(self, graph_store: Store):
        """Test that created_by is set on new records."""
        person = await graph_store.create_person(
            created_by="dcramer",
            name="Sarah",
        )

        assert person.created_by == "dcramer"

    async def test_get_person(self, graph_store: Store):
        """Test getting a person by ID."""
        created = await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
        )

        found = await graph_store.get_person(created.id)

        assert found is not None
        assert found.id == created.id
        assert found.name == "Sarah"

    async def test_find_person_by_name(self, graph_store: Store):
        """Test finding a person by name (globally)."""
        await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
        )

        found = await graph_store.find_person("Sarah")

        assert found is not None
        assert found.name == "Sarah"

    async def test_find_person_by_relationship(self, graph_store: Store):
        """Test finding a person by relationship."""
        await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
            relationship="wife",
        )

        found = await graph_store.find_person("wife")

        assert found is not None
        assert found.name == "Sarah"

    async def test_find_person_by_alias(self, graph_store: Store):
        """Test finding a person by alias."""
        await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
            aliases=["my wife"],
        )

        found = await graph_store.find_person("my wife")

        assert found is not None
        assert found.name == "Sarah"

    async def test_find_person_case_insensitive(self, graph_store: Store):
        """Test that person lookup is case-insensitive."""
        await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
        )

        found = await graph_store.find_person("sarah")
        assert found is not None

        found = await graph_store.find_person("SARAH")
        assert found is not None

    async def test_find_is_global(self, graph_store: Store):
        """Test that find searches all records regardless of creator."""
        await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
        )

        # Can be found without specifying an owner
        found = await graph_store.find_person("Sarah")
        assert found is not None

    async def test_list_all(self, graph_store: Store):
        """Test listing all people, sorted by name."""
        await graph_store.create_person(created_by="user-1", name="Charlie")
        await graph_store.create_person(created_by="user-1", name="Alice")
        await graph_store.create_person(created_by="user-2", name="Bob")

        all_people = await graph_store.list_people()

        assert len(all_people) == 3
        # Ordered by updated_at DESC (most recently created first)
        names = [p.name for p in all_people]
        assert set(names) == {"Alice", "Bob", "Charlie"}

    async def test_update_person(self, graph_store: Store):
        """Test updating person details."""
        person = await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
        )

        updated = await graph_store.update_person(
            person_id=person.id,
            name="Sara",
        )
        assert updated is not None
        assert updated.name == "Sara"

        # Use dedicated add_relationship for relationship changes
        updated = await graph_store.add_relationship(person.id, "wife")
        assert updated is not None
        assert len(updated.relationships) == 1
        assert updated.relationships[0].relationship == "wife"

    async def test_get_all(self, graph_store: Store):
        """Test getting all people across owners."""
        await graph_store.create_person(created_by="user-1", name="Alice")
        await graph_store.create_person(created_by="user-2", name="Bob")

        all_people = await graph_store.get_all_people()

        assert len(all_people) == 2


class TestAliasManagement:
    """Tests for alias management."""

    async def test_add_alias(self, graph_store: Store):
        """Test adding an alias to a person."""
        person = await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
        )

        updated = await graph_store.add_alias(person.id, "honey", "user-1")

        assert updated is not None
        assert any(a.value == "honey" for a in updated.aliases)

    async def test_add_alias_no_duplicates(self, graph_store: Store):
        """Test that duplicate aliases are not added."""
        person = await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
            aliases=["honey"],
        )

        updated = await graph_store.add_alias(person.id, "Honey", "user-1")

        assert updated is not None
        # Should still only have one alias (case-insensitive dedup)
        assert len(updated.aliases) == 1

    async def test_add_alias_anyone_can_add(self, graph_store: Store):
        """Test that any user can add an alias (no owner check)."""
        person = await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
        )

        result = await graph_store.add_alias(person.id, "honey", "user-2")
        assert result is not None
        assert any(a.value == "honey" for a in result.aliases)


class TestAliasProvenance:
    """Tests for alias provenance tracking."""

    async def test_alias_tracks_added_by(self, graph_store: Store):
        """Test that aliases record who added them."""
        person = await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
        )

        updated = await graph_store.add_alias(person.id, "sksembhi", "user-2")

        assert updated is not None
        alias = next(a for a in updated.aliases if a.value == "sksembhi")
        assert alias.added_by == "user-2"
        assert alias.created_at is not None

    async def test_create_aliases_track_added_by(self, graph_store: Store):
        """Test that aliases from create_person() have provenance."""
        person = await graph_store.create_person(
            created_by="dcramer",
            name="Sarah",
            aliases=["my wife"],
        )

        assert person.aliases[0].value == "my wife"
        assert person.aliases[0].added_by == "dcramer"

    async def test_backward_compat_plain_string_aliases(self, graph_store: Store):
        """Test that old plain-string aliases are handled correctly."""
        person = await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
        )
        await graph_store.add_alias(person.id, "my wife", added_by="user-1")
        await graph_store.add_alias(person.id, "honey", added_by="user-1")

        refreshed = await graph_store.get_person(person.id)
        assert refreshed is not None
        assert len(refreshed.aliases) == 2
        alias_values = {a.value for a in refreshed.aliases}
        assert "my wife" in alias_values
        assert "honey" in alias_values


class TestRelationshipProvenance:
    """Tests for relationship provenance tracking."""

    async def test_relationship_tracks_stated_by(self, graph_store: Store):
        """Test that relationships record who stated them."""
        person = await graph_store.create_person(
            created_by="dcramer",
            name="Sarah",
            relationship="wife",
        )

        assert len(person.relationships) == 1
        rc = person.relationships[0]
        assert rc.relationship == "wife"
        assert rc.stated_by == "dcramer"
        assert rc.created_at is not None

    async def test_add_relationship(self, graph_store: Store):
        """Test adding a relationship claim."""
        person = await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
        )

        updated = await graph_store.add_relationship(person.id, "wife", "dcramer")

        assert updated is not None
        assert len(updated.relationships) == 1
        assert updated.relationships[0].relationship == "wife"
        assert updated.relationships[0].stated_by == "dcramer"

    async def test_add_relationship_no_duplicates(self, graph_store: Store):
        """Test that duplicate relationships are not added."""
        person = await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
            relationship="wife",
        )

        updated = await graph_store.add_relationship(person.id, "Wife", "user-2")

        assert updated is not None
        assert len(updated.relationships) == 1

    async def test_add_relationship_string(self, graph_store: Store):
        """Test adding a relationship via the store API."""
        person = await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
            relationship="wife",
        )

        assert len(person.relationships) == 1
        assert person.relationships[0].relationship == "wife"


class TestUsernameMatching:
    """Tests for username matching."""

    async def test_matches_username_by_name(self, graph_store: Store):
        """Test matching by name."""
        person = await graph_store.create_person(
            created_by="user-1",
            name="Bob",
        )

        assert person.matches_username("bob")
        assert person.matches_username("Bob")
        assert not person.matches_username("alice")

    async def test_matches_username_by_alias(self, graph_store: Store):
        """Test matching by alias."""
        person = await graph_store.create_person(
            created_by="user-1",
            name="Bob",
            aliases=["@bobby", "robert"],
        )

        assert person.matches_username("@bobby")
        assert person.matches_username("robert")


class TestResolution:
    """Tests for person resolution."""

    async def test_resolve_existing(self, graph_store: Store):
        """Test resolving an existing person."""
        created = await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
        )

        result = await graph_store.resolve_or_create_person("user-1", "Sarah")

        assert result.person_id == created.id
        assert result.created is False
        assert result.person_name == "Sarah"

    async def test_resolve_creates_new(self, graph_store: Store):
        """Test that resolve creates a new person if not found."""
        result = await graph_store.resolve_or_create_person("user-1", "Bob")

        assert result.created is True
        assert result.person_name == "Bob"

        # Verify it was persisted
        found = await graph_store.get_person(result.person_id)
        assert found is not None
        assert found.name == "Bob"

    async def test_resolve_relationship_term(self, graph_store: Store):
        """Test resolving a relationship term (e.g., 'my wife')."""
        result = await graph_store.resolve_or_create_person(
            "user-1",
            "my wife",
            content_hint="My wife Sarah loves hiking",
        )

        assert result.created is True
        assert result.person_name == "Sarah"

        person = await graph_store.get_person(result.person_id)
        assert person is not None
        assert len(person.relationships) == 1
        assert person.relationships[0].relationship == "wife"

    async def test_resolve_names(self, graph_store: Store):
        """Test resolving person IDs to names."""
        p1 = await graph_store.create_person(created_by="user-1", name="Alice")
        p2 = await graph_store.create_person(created_by="user-1", name="Bob")

        names = await graph_store.resolve_names([p1.id, p2.id])

        assert names == {p1.id: "Alice", p2.id: "Bob"}

    async def test_resolve_names_missing_id(self, graph_store: Store):
        """Test that missing IDs are omitted from results."""
        p1 = await graph_store.create_person(created_by="user-1", name="Alice")

        names = await graph_store.resolve_names([p1.id, "nonexistent"])

        assert names == {p1.id: "Alice"}

    async def test_resolve_global(self, graph_store: Store):
        """Test that resolution finds people created by other users."""
        created = await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
        )

        # A different user resolving the same name should find the existing person
        result = await graph_store.resolve_or_create_person("user-2", "Sarah")

        assert result.person_id == created.id
        assert result.created is False


class TestFindPersonIdsForUsername:
    """Tests for find_person_ids_for_username."""

    async def test_finds_globally(self, graph_store: Store):
        """Test finding person IDs across all creators."""
        p1 = await graph_store.create_person(
            created_by="alice", name="Bob", aliases=["bob"]
        )
        p2 = await graph_store.create_person(
            created_by="carol", name="Bob2", aliases=["bob"]
        )

        ids = await graph_store.find_person_ids_for_username("bob")

        assert ids == {p1.id, p2.id}

    async def test_handles_at_prefix(self, graph_store: Store):
        """Test that @ prefix is stripped during matching."""
        p = await graph_store.create_person(
            created_by="alice", name="Bob", aliases=["bob"]
        )

        ids = await graph_store.find_person_ids_for_username("@bob")

        assert ids == {p.id}

    async def test_returns_empty_for_unknown(self, graph_store: Store):
        """Test that unknown username returns empty set."""
        ids = await graph_store.find_person_ids_for_username("nobody")

        assert ids == set()

    async def test_remaps_merged_ids(self, graph_store: Store):
        """Test that merged person IDs are remapped to primary."""
        p1 = await graph_store.create_person(
            created_by="user-1", name="Sarah", aliases=["sarah"]
        )
        p2 = await graph_store.create_person(
            created_by="user-1", name="Sksembhi", aliases=["sksembhi"]
        )

        # Merge p2 into p1
        await graph_store.merge_people(p1.id, p2.id)

        # Looking up "sksembhi" should return p1's ID (the primary)
        ids = await graph_store.find_person_ids_for_username("sksembhi")
        assert ids == {p1.id}


class TestPersonMerge:
    """Tests for person merging."""

    async def test_merge_combines_aliases(self, graph_store: Store):
        """Test that merge copies aliases from secondary to primary."""
        p1 = await graph_store.create_person(
            created_by="user-1", name="Sarah", aliases=["my wife"]
        )
        p2 = await graph_store.create_person(
            created_by="user-2", name="Sksembhi", aliases=["sksembhi"]
        )

        result = await graph_store.merge_people(p1.id, p2.id)

        assert result is not None
        alias_values = [a.value for a in result.aliases]
        assert "my wife" in alias_values
        assert "sksembhi" in alias_values

    async def test_merge_sets_merged_into(self, graph_store: Store):
        """Test that secondary gets merged_into set."""
        p1 = await graph_store.create_person(created_by="user-1", name="Sarah")
        p2 = await graph_store.create_person(created_by="user-2", name="Sksembhi")

        await graph_store.merge_people(p1.id, p2.id)

        secondary = await graph_store.get_person(p2.id)
        assert secondary is not None
        assert secondary.merged_into == p1.id

    async def test_merge_adds_secondary_name_as_alias(self, graph_store: Store):
        """Test that secondary's name becomes an alias on primary."""
        p1 = await graph_store.create_person(created_by="user-1", name="Sarah")
        p2 = await graph_store.create_person(created_by="user-2", name="Sksembhi")

        result = await graph_store.merge_people(p1.id, p2.id)

        assert result is not None
        alias_values = [a.value for a in result.aliases]
        assert "Sksembhi" in alias_values

    async def test_merge_does_not_duplicate_same_name(self, graph_store: Store):
        """Test that merge doesn't add name as alias if same as primary."""
        p1 = await graph_store.create_person(created_by="user-1", name="Sarah")
        p2 = await graph_store.create_person(created_by="user-2", name="Sarah")

        result = await graph_store.merge_people(p1.id, p2.id)

        assert result is not None
        # Should not have "Sarah" as an alias since it matches primary name
        alias_values = [a.value.lower() for a in result.aliases]
        assert "sarah" not in alias_values

    async def test_merged_person_excluded_from_list(self, graph_store: Store):
        """Test that merged records don't appear in list_people."""
        p1 = await graph_store.create_person(created_by="user-1", name="Sarah")
        p2 = await graph_store.create_person(created_by="user-2", name="Sksembhi")

        await graph_store.merge_people(p1.id, p2.id)

        people = await graph_store.list_people()
        assert len(people) == 1
        assert people[0].id == p1.id

    async def test_merged_person_excluded_from_find(self, graph_store: Store):
        """Test that merged records don't appear in find results."""
        p1 = await graph_store.create_person(created_by="user-1", name="Sarah")
        p2 = await graph_store.create_person(
            created_by="user-2", name="Sksembhi", aliases=["sks"]
        )

        await graph_store.merge_people(p1.id, p2.id)

        # Finding by secondary's name should not return the merged record
        found = await graph_store.find_person("Sksembhi")
        # It should find p1 via the alias added during merge
        assert found is not None
        assert found.id == p1.id

    async def test_resolve_follows_merge_chain(self, graph_store: Store):
        """Test that resolve_or_create_person follows merge chain."""
        p1 = await graph_store.create_person(created_by="user-1", name="Sarah")
        p2 = await graph_store.create_person(
            created_by="user-2", name="Sksembhi", aliases=["sks"]
        )

        await graph_store.merge_people(p1.id, p2.id)

        # Resolving by an alias that exists on primary (from merge) should return primary
        result = await graph_store.resolve_or_create_person("user-3", "Sksembhi")
        assert result.person_id == p1.id
        assert result.created is False

    async def test_merge_copies_relationships(self, graph_store: Store):
        """Test that merge copies relationships from secondary."""
        p1 = await graph_store.create_person(created_by="user-1", name="Sarah")
        p2 = await graph_store.create_person(
            created_by="user-2", name="Sksembhi", relationship="wife"
        )

        result = await graph_store.merge_people(p1.id, p2.id)

        assert result is not None
        assert len(result.relationships) == 1
        assert result.relationships[0].relationship == "wife"

    async def test_merge_returns_none_for_missing(self, graph_store: Store):
        """Test that merge returns None if either ID is missing."""
        p1 = await graph_store.create_person(created_by="user-1", name="Sarah")

        result = await graph_store.merge_people(p1.id, "nonexistent")
        assert result is None

    async def test_merge_refuses_already_merged_secondary(self, graph_store: Store):
        """Merge should refuse if secondary is already merged (prevents chain corruption)."""
        p1 = await graph_store.create_person(created_by="user-1", name="Sarah")
        p2 = await graph_store.create_person(created_by="user-1", name="Sksembhi")
        p3 = await graph_store.create_person(created_by="user-1", name="Charlie")

        # Merge p2 into p1
        await graph_store.merge_people(p1.id, p2.id)

        # Now try to merge p2 into p3 — should be refused
        result = await graph_store.merge_people(p3.id, p2.id)
        assert result is None

        # p2 should still point to p1, not p3
        p2_after = await graph_store.get_person(p2.id)
        assert p2_after is not None
        assert p2_after.merged_into == p1.id


class TestCacheBehavior:
    """Tests for cache invalidation."""

    async def test_cache_invalidates_after_create(self, graph_store: Store):
        """Test that cache is invalidated after creating a person."""
        await graph_store.create_person(created_by="user-1", name="Alice")
        people1 = await graph_store.get_all_people()
        assert len(people1) == 1

        await graph_store.create_person(created_by="user-1", name="Bob")
        people2 = await graph_store.get_all_people()
        assert len(people2) == 2

    async def test_cache_invalidates_after_update(self, graph_store: Store):
        """Test that cache is invalidated after update."""
        person = await graph_store.create_person(created_by="user-1", name="Alice")

        await graph_store.update_person(person_id=person.id, name="Alicia")

        refreshed = await graph_store.get_person(person.id)
        assert refreshed is not None
        assert refreshed.name == "Alicia"


class TestSerialization:
    """Tests for PersonEntry construction."""

    def test_construction(self):
        """Test that PersonEntry can be constructed with all fields."""
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

        assert entry.id == "test-123"
        assert entry.created_by == "dcramer"
        assert entry.name == "Sarah"
        assert len(entry.relationships) == 1
        assert entry.relationships[0].relationship == "wife"
        assert entry.relationships[0].stated_by == "dcramer"
        assert len(entry.aliases) == 2
        assert entry.aliases[0].value == "my wife"
        assert entry.aliases[1].value == "sksembhi"


class TestNormalizeReference:
    """Tests for _normalize_reference prefix stripping."""

    def test_strips_my_prefix(self):
        assert Store._normalize_reference("my wife") == "wife"

    def test_strips_the_prefix(self):
        assert Store._normalize_reference("the boss") == "boss"

    def test_strips_at_prefix(self):
        assert Store._normalize_reference("@dcramer") == "dcramer"

    def test_strips_my_then_at(self):
        """After stripping 'my ', '@dcramer' should also strip '@'."""
        assert Store._normalize_reference("my @dcramer") == "dcramer"

    def test_no_prefix(self):
        assert Store._normalize_reference("sarah") == "sarah"


class TestParsePersonReference:
    """Tests for _parse_person_reference relationship filtering."""

    def test_valid_relationship_term(self, graph_store: Store):
        name, rel = graph_store._parse_person_reference("my wife")
        assert rel == "wife"

    def test_invalid_relationship_term_returns_none(self, graph_store: Store):
        """Non-relationship terms like 'cat' should not be stored as relationships."""
        name, rel = graph_store._parse_person_reference("my cat")
        assert rel is None
        # "my cat" is not a recognized relationship, so full string is title-cased
        assert name == "My Cat"

    def test_plain_name_no_relationship(self, graph_store: Store):
        name, rel = graph_store._parse_person_reference("Sarah")
        assert rel is None
        assert name == "Sarah"


class TestUpdateProvenance:
    """Tests for add_relationship/add_alias provenance tracking."""

    async def test_add_relationship_with_provenance(self, graph_store: Store):
        person = await graph_store.create_person(created_by="user-1", name="Sarah")

        updated = await graph_store.add_relationship(
            person.id,
            "wife",
            stated_by="dcramer",
        )

        assert updated is not None
        assert len(updated.relationships) == 1
        rc = updated.relationships[0]
        assert rc.relationship == "wife"
        assert rc.stated_by == "dcramer"
        assert rc.created_at is not None

    async def test_add_aliases_with_provenance(self, graph_store: Store):
        person = await graph_store.create_person(created_by="user-1", name="Sarah")

        for alias in ["sksembhi", "honey"]:
            updated = await graph_store.add_alias(person.id, alias, added_by="dcramer")

        assert updated is not None
        assert len(updated.aliases) == 2
        for alias in updated.aliases:
            assert alias.added_by == "dcramer"
            assert alias.created_at is not None

    async def test_add_relationship_without_stated_by(self, graph_store: Store):
        """Provenance fields are None when stated_by is not provided."""
        person = await graph_store.create_person(created_by="user-1", name="Sarah")

        updated = await graph_store.add_relationship(person.id, "wife")

        assert updated is not None
        assert updated.relationships[0].stated_by is None


class TestPersonDelete:
    """Tests for person deletion."""

    async def test_delete_person(self, graph_store: Store):
        """Test deleting a person."""
        person = await graph_store.create_person(created_by="user-1", name="Sarah")

        result = await graph_store.delete_person(person.id)
        assert result is True

        found = await graph_store.get_person(person.id)
        assert found is None

    async def test_delete_nonexistent(self, graph_store: Store):
        """Test deleting a person that doesn't exist."""
        result = await graph_store.delete_person("nonexistent")
        assert result is False

    async def test_delete_clears_merged_into(self, graph_store: Store):
        """Test that deleting a person clears merged_into references."""
        p1 = await graph_store.create_person(created_by="user-1", name="Sarah")
        p2 = await graph_store.create_person(created_by="user-1", name="Sksembhi")

        await graph_store.merge_people(p1.id, p2.id)

        # Delete the primary — secondary's merged_into should be cleared
        await graph_store.delete_person(p1.id)

        p2_after = await graph_store.get_person(p2.id)
        assert p2_after is not None
        assert p2_after.merged_into is None

    async def test_delete_removes_from_list(self, graph_store: Store):
        """Test that deleted person is gone from list_people."""
        p1 = await graph_store.create_person(created_by="user-1", name="Alice")
        p2 = await graph_store.create_person(created_by="user-1", name="Bob")

        await graph_store.delete_person(p1.id)

        people = await graph_store.list_people()
        assert len(people) == 1
        assert people[0].id == p2.id


class TestMultiWordNameExtraction:
    """Tests for multi-word name extraction from content."""

    def test_extracts_two_word_name(self):
        """Test extracting 'Sarah Jane' from 'my wife Sarah Jane loves hiking'."""
        result = Store._extract_name_from_content(
            "My wife Sarah Jane loves hiking", "wife"
        )
        assert result == "Sarah Jane"

    def test_extracts_single_word_name(self):
        """Test extracting single-word name still works."""
        result = Store._extract_name_from_content("My wife Sarah loves hiking", "wife")
        assert result == "Sarah"

    def test_extracts_name_from_is_named(self):
        """Test 'wife is named Sarah Jane' pattern."""
        result = Store._extract_name_from_content("My wife is named Sarah Jane", "wife")
        assert result == "Sarah Jane"

    def test_extracts_name_from_possessive(self):
        """Test 'wife's name is Sarah Jane' pattern."""
        result = Store._extract_name_from_content(
            "My wife's name is Sarah Jane", "wife"
        )
        assert result == "Sarah Jane"


class TestFuzzyResolution:
    """Tests for fuzzy resolution graceful degradation (no LLM mocking).

    Actual fuzzy matching quality is tested by the identity eval
    (evals/test_identity.py) which uses real LLM calls.
    """

    async def test_no_llm_degrades_to_create(self, graph_store: Store):
        """Without LLM configured, fuzzy match is skipped and a new person is created."""
        await graph_store.create_person(
            created_by="user-1",
            name="Sukhpreet Sembhi",
        )

        # No set_llm() call — LLM is None
        result = await graph_store.resolve_or_create_person("user-1", "Sukhpreet")

        assert result.created is True
        assert result.person_name == "Sukhpreet"

    async def test_llm_error_degrades_to_create(self, graph_store: Store):
        """LLM exception degrades gracefully to creating a new person."""
        await graph_store.create_person(
            created_by="user-1",
            name="Sukhpreet Sembhi",
        )

        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = RuntimeError("API error")
        graph_store.set_llm(mock_llm, "test-model")

        result = await graph_store.resolve_or_create_person("user-1", "Sukhpreet")

        assert result.created is True


class TestRelationshipStatedBy:
    """Tests for relationship_stated_by override (Bug 1 fix)."""

    async def test_create_with_relationship_stated_by(self, graph_store: Store):
        """stated_by uses the override, not created_by."""
        person = await graph_store.create_person(
            created_by="123456789",
            name="Sarah",
            relationship="wife",
            relationship_stated_by="dcramer",
        )

        assert person.relationships[0].stated_by == "dcramer"
        assert person.created_by == "123456789"

    async def test_create_relationship_stated_by_fallback(self, graph_store: Store):
        """Falls back to created_by when relationship_stated_by not provided."""
        person = await graph_store.create_person(
            created_by="dcramer",
            name="Sarah",
            relationship="wife",
        )

        assert person.relationships[0].stated_by == "dcramer"

    async def test_resolve_or_create_passes_relationship_stated_by(
        self, graph_store: Store
    ):
        """relationship_stated_by flows through to created person."""
        result = await graph_store.resolve_or_create_person(
            created_by="123456789",
            reference="my wife",
            content_hint="My wife Sarah loves hiking",
            relationship_stated_by="dcramer",
        )

        assert result.created is True
        person = await graph_store.get_person(result.person_id)
        assert person is not None
        assert person.relationships[0].stated_by == "dcramer"


class TestOwnerNameMatchers:
    """Tests for build_owner_matchers name matching (Bug 3 fix).

    Prefers false positives (dropping a valid subject) over false negatives
    (creating a duplicate person entry for the speaker).
    """

    def test_first_name_of_multi_word_matches(self):
        """First name of multi-word name matches as owner."""
        matchers = build_owner_matchers(["David Cramer", "dcramer"])
        assert is_owner_name("David", matchers) is True

    def test_last_name_of_multi_word_matches(self):
        """Last name of multi-word name matches as owner."""
        matchers = build_owner_matchers(["David Cramer"])
        assert is_owner_name("Cramer", matchers) is True

    def test_single_word_name_no_parts(self):
        """Single-word name does NOT add parts — exact-only matching."""
        matchers = build_owner_matchers(["dcramer"])
        # "dcramer" is exact match
        assert is_owner_name("dcramer", matchers) is True
        # No parts were added for single-word names
        assert len(matchers.parts) == 0

    def test_prefix_of_name_part_matches(self):
        """Prefix of a name part matches (e.g., Davi → David)."""
        matchers = build_owner_matchers(["David Cramer"])
        # "davi" is a prefix of part "david"
        assert is_owner_name("Davi", matchers) is True
        # "dave" is NOT a prefix of "david" — true nicknames aren't caught
        assert is_owner_name("Dave", matchers) is False

    def test_subject_starting_with_name_part_matches(self):
        """Subject starting with a name part matches (e.g., 'David C.' → 'david')."""
        matchers = build_owner_matchers(["David Cramer"])
        assert is_owner_name("David C.", matchers) is True

    def test_short_subjects_skip_prefix_check(self):
        """Very short subjects (< 3 chars) don't trigger prefix matching."""
        matchers = build_owner_matchers(["David Cramer"])
        assert is_owner_name("Da", matchers) is False

    def test_username_not_split(self):
        """Username is not split into parts."""
        matchers = build_owner_matchers(["dcramer"])
        assert is_owner_name("dcr", matchers) is False

    def test_full_name_exact_match(self):
        """Full name still matches exactly."""
        matchers = build_owner_matchers(["David Cramer"])
        assert is_owner_name("David Cramer", matchers) is True

    def test_at_prefix_stripped(self):
        """@ prefix is stripped before matching."""
        matchers = build_owner_matchers(["dcramer"])
        assert is_owner_name("@dcramer", matchers) is True

    def test_unrelated_name_does_not_match(self):
        """Completely unrelated names don't match."""
        matchers = build_owner_matchers(["David Cramer", "dcramer"])
        assert is_owner_name("Sarah", matchers) is False
        assert is_owner_name("Bob Smith", matchers) is False

    def test_short_name_parts_excluded(self):
        """Name parts shorter than 3 chars are excluded from part matching."""
        matchers = build_owner_matchers(["Li Wei"])
        # "li" is too short (< 3) to be a part
        assert "li" not in matchers.parts
        assert "wei" in matchers.parts


class TestContentAwareFuzzyFind:
    """Tests for content_hint and speaker passing in fuzzy matching."""

    async def test_fuzzy_find_passes_content_hint(self, graph_store: Store):
        """Verify content_hint is forwarded to _fuzzy_find via resolve_or_create_person."""
        await graph_store.create_person(
            created_by="user-1",
            name="Wife",
            relationship="wife",
        )

        # Mock LLM to return matching person ID
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.get_text.return_value = "NONE"
        mock_llm.complete.return_value = mock_response
        graph_store.set_llm(mock_llm, "test-model")

        await graph_store.resolve_or_create_person(
            "user-1", "Sarah", content_hint="my wife Sarah's birthday is March 15"
        )

        # Verify LLM was called with a prompt that includes the content hint
        call_args = mock_llm.complete.call_args
        prompt = call_args.kwargs["messages"][0].content
        assert "my wife Sarah's birthday is March 15" in prompt

    async def test_fuzzy_find_without_content_hint(self, graph_store: Store):
        """Verify _fuzzy_find works without content_hint (no Context line)."""
        await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
        )

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.get_text.return_value = "NONE"
        mock_llm.complete.return_value = mock_response
        graph_store.set_llm(mock_llm, "test-model")

        await graph_store.resolve_or_create_person("user-1", "Bob")

        call_args = mock_llm.complete.call_args
        prompt = call_args.kwargs["messages"][0].content
        assert "Context:" not in prompt

    async def test_fuzzy_find_includes_speaker(self, graph_store: Store):
        """Speaker is included in the fuzzy match prompt."""
        await graph_store.create_person(
            created_by="user-1",
            name="Wife",
            relationship="wife",
            relationship_stated_by="dcramer",
        )

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.get_text.return_value = "NONE"
        mock_llm.complete.return_value = mock_response
        graph_store.set_llm(mock_llm, "test-model")

        await graph_store.resolve_or_create_person(
            "user-1",
            "Sarah",
            content_hint="Sarah's birthday is March 15",
            relationship_stated_by="dcramer",
        )

        call_args = mock_llm.complete.call_args
        prompt = call_args.kwargs["messages"][0].content
        assert 'Speaker: "dcramer"' in prompt

    async def test_fuzzy_find_includes_stated_by_on_relationships(
        self, graph_store: Store
    ):
        """Relationship stated_by provenance appears in the fuzzy match prompt."""
        await graph_store.create_person(
            created_by="user-1",
            name="Wife",
            relationship="wife",
            relationship_stated_by="dcramer",
        )

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.get_text.return_value = "NONE"
        mock_llm.complete.return_value = mock_response
        graph_store.set_llm(mock_llm, "test-model")

        await graph_store.resolve_or_create_person(
            "user-1", "Sarah", relationship_stated_by="dcramer"
        )

        call_args = mock_llm.complete.call_args
        prompt = call_args.kwargs["messages"][0].content
        assert "stated by dcramer" in prompt

    async def test_fuzzy_find_speaker_falls_back_to_created_by(
        self, graph_store: Store
    ):
        """Without relationship_stated_by, speaker falls back to created_by."""
        await graph_store.create_person(created_by="user-1", name="Sarah")

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.get_text.return_value = "NONE"
        mock_llm.complete.return_value = mock_response
        graph_store.set_llm(mock_llm, "test-model")

        await graph_store.resolve_or_create_person("user-42", "Bob")

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
        assert Store._heuristic_match(a, b) is True

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
        assert Store._heuristic_match(a, b) is True

    def test_name_substring(self):
        """Name substring should match."""
        a = PersonEntry(id="a", name="Sukhpreet")
        b = PersonEntry(id="b", name="Sukhpreet Sembhi")
        assert Store._heuristic_match(a, b) is True

    def test_first_name_overlap(self):
        """First name matching another's full name should match."""
        a = PersonEntry(id="a", name="Sarah")
        b = PersonEntry(id="b", name="Sarah Jane")
        assert Store._heuristic_match(a, b) is True

    def test_skip_both_self_different_users(self):
        """Should skip pairs where both have relationship self from different users."""
        a = PersonEntry(
            id="a",
            created_by="user-1",
            name="David",
            relationships=[RelationshipClaim(relationship="self")],
        )
        b = PersonEntry(
            id="b",
            created_by="user-2",
            name="David Cramer",
            relationships=[RelationshipClaim(relationship="self")],
        )
        assert Store._heuristic_match(a, b) is False

    def test_both_self_same_user_matches(self):
        """Same-user self-persons should match (duplicate self-records)."""
        a = PersonEntry(
            id="a",
            created_by="user-1",
            name="David",
            relationships=[RelationshipClaim(relationship="self")],
        )
        b = PersonEntry(
            id="b",
            created_by="user-1",
            name="David Cramer",
            relationships=[RelationshipClaim(relationship="self")],
        )
        assert Store._heuristic_match(a, b) is True

    def test_no_match_unrelated(self):
        """Unrelated people should not match."""
        a = PersonEntry(id="a", name="Sarah")
        b = PersonEntry(id="b", name="Bob")
        assert Store._heuristic_match(a, b) is False

    def test_short_names_no_substring(self):
        """Names shorter than 3 chars should not trigger substring match."""
        a = PersonEntry(id="a", name="Al")
        b = PersonEntry(id="b", name="Alice")
        assert Store._heuristic_match(a, b) is False

    def test_two_multiword_names_shared_first_name_no_match(self):
        """Two multi-word names sharing only a first name should NOT match."""
        a = PersonEntry(id="a", name="David Chen")
        b = PersonEntry(id="b", name="David Cramer")
        assert Store._heuristic_match(a, b) is False

    def test_single_word_vs_multiword_shared_name_matches(self):
        """Single-word name matching a part of multi-word name SHOULD match."""
        a = PersonEntry(id="a", name="David")
        b = PersonEntry(id="b", name="David Cramer")
        assert Store._heuristic_match(a, b) is True

    def test_shared_alias_overrides_self_check(self):
        """Shared alias on two self-persons from different users should match."""
        a = PersonEntry(
            id="a",
            created_by="user-1",
            name="David Cramer",
            relationships=[RelationshipClaim(relationship="self")],
            aliases=[AliasEntry(value="notzeeg")],
        )
        b = PersonEntry(
            id="b",
            created_by="user-2",
            name="David Cramer",
            relationships=[RelationshipClaim(relationship="self")],
            aliases=[AliasEntry(value="notzeeg")],
        )
        assert Store._heuristic_match(a, b) is True

    def test_name_matches_alias(self):
        """Person name matching another's alias should match."""
        a = PersonEntry(id="a", name="Sksembhi")
        b = PersonEntry(
            id="b",
            name="Sukhpreet Sembhi",
            aliases=[AliasEntry(value="sksembhi")],
        )
        assert Store._heuristic_match(a, b) is True

    def test_alias_at_prefix_normalized(self):
        """@-prefixed alias should match non-prefixed alias after normalization."""
        a = PersonEntry(
            id="a",
            name="David Cramer",
            aliases=[AliasEntry(value="@notzeeg")],
        )
        b = PersonEntry(
            id="b",
            name="Notzeeg",
            aliases=[AliasEntry(value="notzeeg")],
        )
        assert Store._heuristic_match(a, b) is True


class TestFindDedupCandidates:
    """Tests for find_dedup_candidates."""

    async def test_no_llm_returns_empty(self, graph_store: Store):
        """Without LLM, dedup returns empty list."""
        await graph_store.create_person(created_by="user-1", name="Sarah")
        p2 = await graph_store.create_person(created_by="user-1", name="Sarah Jane")

        result = await graph_store.find_dedup_candidates([p2.id])
        assert result == []

    async def test_finds_heuristic_match_with_llm_yes(self, graph_store: Store):
        """Heuristic match confirmed by LLM returns merge candidate."""
        p1 = await graph_store.create_person(
            created_by="user-1",
            name="Sarah",
            relationship="wife",
        )
        p2 = await graph_store.create_person(created_by="user-1", name="Sarah Jane")

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.get_text.return_value = "YES"
        mock_llm.complete.return_value = mock_response
        graph_store.set_llm(mock_llm, "test-model")

        result = await graph_store.find_dedup_candidates([p2.id])
        assert len(result) == 1
        # p1 has more data (relationship), so it's primary
        assert result[0] == (p1.id, p2.id)

    async def test_heuristic_match_with_llm_no(self, graph_store: Store):
        """Heuristic match rejected by LLM returns no candidates."""
        await graph_store.create_person(created_by="user-1", name="Sarah")
        p2 = await graph_store.create_person(created_by="user-1", name="Sarah Connor")

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.get_text.return_value = "NO"
        mock_llm.complete.return_value = mock_response
        graph_store.set_llm(mock_llm, "test-model")

        result = await graph_store.find_dedup_candidates([p2.id])
        assert result == []

    async def test_no_heuristic_match_no_llm_call(self, graph_store: Store):
        """No heuristic match means no LLM verification call."""
        await graph_store.create_person(created_by="user-1", name="Sarah")
        p2 = await graph_store.create_person(created_by="user-1", name="Bob")

        mock_llm = AsyncMock()
        graph_store.set_llm(mock_llm, "test-model")

        result = await graph_store.find_dedup_candidates([p2.id])
        assert result == []
        mock_llm.complete.assert_not_called()

    async def test_all_ids_no_duplicate_pairs(self, graph_store: Store):
        """When all IDs are passed (doctor mode), each pair is checked only once."""
        p1 = await graph_store.create_person(created_by="user-1", name="Sarah")
        p2 = await graph_store.create_person(created_by="user-1", name="Sarah Jane")

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.get_text.return_value = "YES"
        mock_llm.complete.return_value = mock_response
        graph_store.set_llm(mock_llm, "test-model")

        # Pass BOTH IDs (doctor mode) — should NOT generate duplicate pairs
        result = await graph_store.find_dedup_candidates([p1.id, p2.id])
        assert len(result) == 1
        # LLM should be called exactly once, not twice
        assert mock_llm.complete.call_count == 1

    async def test_exclude_self_allows_non_self_vs_self(self, graph_store: Store):
        """exclude_self=True allows matching non-self person against self person."""
        await graph_store.create_person(
            created_by="user-1",
            name="David Cramer",
            relationship="self",
        )
        p2 = await graph_store.create_person(created_by="user-2", name="David")

        graph_store.set_llm(_make_mock_llm("YES"), "test-model")

        # With exclude_self=True, non-self "David" CAN match self "David Cramer"
        result = await graph_store.find_dedup_candidates([p2.id], exclude_self=True)
        assert len(result) == 1

    async def test_exclude_self_blocks_both_self(self, graph_store: Store):
        """exclude_self=True blocks when BOTH persons have 'self' relationship."""
        await graph_store.create_person(
            created_by="user-1",
            name="David Cramer",
            relationship="self",
        )
        p2 = await graph_store.create_person(
            created_by="user-2",
            name="David",
            relationship="self",
        )

        mock_llm = AsyncMock()
        graph_store.set_llm(mock_llm, "test-model")

        result = await graph_store.find_dedup_candidates([p2.id], exclude_self=True)
        assert result == []
        mock_llm.complete.assert_not_called()

    async def test_exclude_self_false_allows_self_person(self, graph_store: Store):
        """exclude_self=False (default) allows matching against self-persons."""
        await graph_store.create_person(
            created_by="user-1",
            name="David Cramer",
            relationship="self",
        )
        p2 = await graph_store.create_person(created_by="user-2", name="David")

        graph_store.set_llm(_make_mock_llm("YES"), "test-model")

        # Without exclude_self (default=False), should find candidates
        result = await graph_store.find_dedup_candidates([p2.id])
        assert len(result) == 1

    async def test_three_duplicates_cluster_into_minimal_merges(
        self, graph_store: Store
    ):
        """Three records for the same person should produce exactly 2 merges
        into a single primary, not 3 redundant pairwise merges."""
        p1 = await graph_store.create_person(created_by="user-1", name="Evan Purkhiser")
        p2 = await graph_store.create_person(created_by="user-1", name="Evan Purkhiser")
        p3 = await graph_store.create_person(
            created_by="user-1",
            name="Evan Purkhiser",
            relationship="coworker",
        )

        graph_store.set_llm(_make_mock_llm("YES"), "test-model")

        result = await graph_store.find_dedup_candidates([p1.id, p2.id, p3.id])

        # Should produce exactly 2 merges (N-1 for a cluster of 3)
        assert len(result) == 2
        # All merges should target the same primary (p3 has most data)
        primaries = {r[0] for r in result}
        assert primaries == {p3.id}
        # Secondaries should be the other two
        secondaries = {r[1] for r in result}
        assert secondaries == {p1.id, p2.id}

    async def test_four_duplicates_cluster_into_minimal_merges(
        self, graph_store: Store
    ):
        """Four records for the same person should produce exactly 3 merges."""
        p1 = await graph_store.create_person(created_by="user-1", name="Sukhpreet")
        p2 = await graph_store.create_person(
            created_by="user-1", name="Sukhpreet Sembhi"
        )
        p3 = await graph_store.create_person(created_by="user-1", name="Sukhpreet")
        p4 = await graph_store.create_person(
            created_by="user-1",
            name="Sukhpreet Sembhi",
            relationship="friend",
        )

        graph_store.set_llm(_make_mock_llm("YES"), "test-model")

        result = await graph_store.find_dedup_candidates([p1.id, p2.id, p3.id, p4.id])

        # Exactly 3 merges (N-1 for cluster of 4), all into same primary
        assert len(result) == 3
        primaries = {r[0] for r in result}
        assert primaries == {p4.id}

    async def test_two_separate_clusters(self, graph_store: Store):
        """Two distinct groups of duplicates should produce independent clusters."""
        # Cluster A: two Sarahs
        sa1 = await graph_store.create_person(
            created_by="user-1", name="Sarah", relationship="wife"
        )
        sa2 = await graph_store.create_person(created_by="user-1", name="Sarah Jane")
        # Cluster B: two Bobs
        sb1 = await graph_store.create_person(
            created_by="user-1", name="Bob", relationship="brother"
        )
        sb2 = await graph_store.create_person(created_by="user-1", name="Bobby")

        graph_store.set_llm(_make_mock_llm("YES"), "test-model")

        result = await graph_store.find_dedup_candidates(
            [sa1.id, sa2.id, sb1.id, sb2.id]
        )

        # Two clusters of 2 -> 2 merges total
        assert len(result) == 2
        primaries = {r[0] for r in result}
        assert primaries == {sa1.id, sb1.id}

    async def test_same_user_self_persons_detected(self, graph_store: Store):
        """Duplicate self-persons from the same user should be detected as candidates."""
        p1 = await graph_store.create_person(
            created_by="user-1",
            name="David Cramer",
            relationship="self",
        )
        p2 = await graph_store.create_person(
            created_by="user-1",
            name="David Cramer",
            relationship="self",
        )

        graph_store.set_llm(_make_mock_llm("YES"), "test-model")

        result = await graph_store.find_dedup_candidates([p1.id, p2.id])
        assert len(result) == 1

    async def test_different_user_self_persons_not_detected(self, graph_store: Store):
        """Self-persons from different users should NOT be detected as candidates."""
        p1 = await graph_store.create_person(
            created_by="user-1",
            name="David Cramer",
            relationship="self",
        )
        p2 = await graph_store.create_person(
            created_by="user-2",
            name="David Cramer",
            relationship="self",
        )

        mock_llm = _make_mock_llm("YES")
        graph_store.set_llm(mock_llm, "test-model")

        result = await graph_store.find_dedup_candidates([p1.id, p2.id])
        assert result == []
        mock_llm.complete.assert_not_called()


class TestAutoRemapOnMerge:
    """Tests for automatic memory remap on merge."""

    async def test_merge_calls_remap(self, graph_store: Store):
        """Merge should remap subject_person_ids in memories from secondary to primary."""
        p1 = await graph_store.create_person(created_by="user-1", name="Sarah")
        p2 = await graph_store.create_person(created_by="user-1", name="Sksembhi")

        # Create a memory with subject_person_ids pointing to the secondary
        await graph_store.add_memory(
            content="Sksembhi likes coffee",
            owner_user_id="user-1",
            subject_person_ids=[p2.id],
        )

        await graph_store.merge_people(p1.id, p2.id)

        # Verify that memories now point to the primary
        memories = await graph_store.get_all_memories()
        for mem in memories:
            if mem.subject_person_ids:
                assert p2.id not in mem.subject_person_ids
                assert p1.id in mem.subject_person_ids

    async def test_merge_without_memories(self, graph_store: Store):
        """Merge should work fine when no memories reference the secondary."""
        p1 = await graph_store.create_person(created_by="user-1", name="Sarah")
        p2 = await graph_store.create_person(created_by="user-1", name="Sksembhi")

        result = await graph_store.merge_people(p1.id, p2.id)
        assert result is not None
        assert result.id == p1.id


class TestSpeakerScopedResolution:
    """Tests for speaker-scoped resolution via find_person_for_speaker."""

    async def test_alice_and_bob_separate_sarahs(self, graph_store: Store):
        """Alice's wife Sarah and Bob's coworker Sarah are distinct persons."""
        # Alice creates "my wife Sarah"
        alice_result = await graph_store.resolve_or_create_person(
            created_by="alice-id",
            reference="my wife",
            content_hint="My wife Sarah likes hiking",
            relationship_stated_by="alice-id",
        )

        # Bob creates "my coworker Sarah"
        bob_result = await graph_store.resolve_or_create_person(
            created_by="bob-id",
            reference="my coworker",
            content_hint="My coworker Sarah is great with databases",
            relationship_stated_by="bob-id",
        )

        # Should be two distinct persons
        assert alice_result.person_id != bob_result.person_id

    async def test_find_for_speaker_returns_connected_person(self, graph_store: Store):
        """find_person_for_speaker returns person connected to speaker via KNOWS edge."""
        # Create Sarah with Alice's relationship
        sarah = await graph_store.create_person(
            created_by="alice-id",
            name="Sarah",
            relationship="wife",
            relationship_stated_by="alice-id",
        )

        result = await graph_store.find_person_for_speaker("Sarah", "alice-id")
        assert result is not None
        assert result.id == sarah.id

    async def test_find_for_speaker_by_relationship_term(self, graph_store: Store):
        """find_person_for_speaker matches by relationship term (e.g., 'wife')."""
        sarah = await graph_store.create_person(
            created_by="alice-id",
            name="Sarah",
            relationship="wife",
            relationship_stated_by="alice-id",
        )

        result = await graph_store.find_person_for_speaker("wife", "alice-id")
        assert result is not None
        assert result.id == sarah.id

    async def test_find_for_speaker_different_speaker_no_match(
        self, graph_store: Store
    ):
        """find_person_for_speaker returns None when speaker has no edges to any Sarah."""
        await graph_store.create_person(
            created_by="alice-id",
            name="Sarah",
            relationship="wife",
            relationship_stated_by="alice-id",
        )

        # Charlie has no edges to any Sarah
        result = await graph_store.find_person_for_speaker("Sarah", "charlie-id")
        assert result is None

    async def test_find_for_speaker_falls_through_to_global(self, graph_store: Store):
        """When speaker has no match, resolve_or_create_person falls through to global find."""
        sarah = await graph_store.create_person(
            created_by="alice-id",
            name="Sarah",
        )

        # Charlie resolving "Sarah" with no speaker edges falls to global find
        result = await graph_store.resolve_or_create_person(
            created_by="charlie-id",
            reference="Sarah",
            relationship_stated_by="charlie-id",
        )

        # Should find the globally existing Sarah
        assert result.person_id == sarah.id
        assert result.created is False

    async def test_find_for_speaker_scopes_by_alias_added_by(self, graph_store: Store):
        """find_person_for_speaker matches via ALIAS edge (added_by)."""
        sarah = await graph_store.create_person(
            created_by="alice-id",
            name="Sarah",
            aliases=["sks"],
        )

        result = await graph_store.find_person_for_speaker("sks", "alice-id")
        assert result is not None
        assert result.id == sarah.id

    async def test_resolve_or_create_prefers_speaker_match(self, graph_store: Store):
        """resolve_or_create_person prefers speaker-scoped match over global."""
        # Create two Sarahs - one connected to Alice, one to Bob
        alice_sarah = await graph_store.create_person(
            created_by="alice-id",
            name="Sarah",
            relationship="wife",
            relationship_stated_by="alice-id",
            aliases=["alice-sarah"],
        )
        await graph_store.create_person(
            created_by="bob-id",
            name="Sarah B",
            relationship="coworker",
            relationship_stated_by="bob-id",
            aliases=["Sarah"],
        )

        # Alice resolving "Sarah" should find her Sarah (via KNOWS edge)
        result = await graph_store.resolve_or_create_person(
            created_by="alice-id",
            reference="Sarah",
            relationship_stated_by="alice-id",
        )
        assert result.person_id == alice_sarah.id


class TestNoUsernameFallback:
    """Tests for no-username user fallback (numeric ID as alias)."""

    async def test_self_person_with_no_username_gets_user_id_alias(
        self, graph_store: Store
    ):
        """Self-person created with no username uses numeric user_id as alias."""
        from ash.core.agent import Agent

        # Create a minimal agent with graph_store as _people
        agent = Agent.__new__(Agent)
        agent._people = graph_store

        await agent._ensure_self_person(
            user_id="123456789",
            username="",
            display_name="David Cramer",
        )

        # Should have created a person with user_id as alias
        people = await graph_store.list_people()
        assert len(people) == 1
        assert people[0].name == "David Cramer"
        alias_values = [a.value for a in people[0].aliases]
        assert "123456789" in alias_values

    async def test_no_username_user_id_resolves(self, graph_store: Store):
        """find_person_ids_for_username with numeric ID resolves after no-username create."""
        from ash.core.agent import Agent

        agent = Agent.__new__(Agent)
        agent._people = graph_store

        await agent._ensure_self_person(
            user_id="123456789",
            username="",
            display_name="David Cramer",
        )

        ids = await graph_store.find_person_ids_for_username("123456789")
        assert len(ids) == 1

    async def test_self_person_with_username_uses_username_alias(
        self, graph_store: Store
    ):
        """Self-person created with username uses username as alias (not user_id)."""
        from ash.core.agent import Agent

        agent = Agent.__new__(Agent)
        agent._people = graph_store

        await agent._ensure_self_person(
            user_id="123456789",
            username="notzeeg",
            display_name="David Cramer",
        )

        people = await graph_store.list_people()
        assert len(people) == 1
        alias_values = [a.value for a in people[0].aliases]
        assert "notzeeg" in alias_values
        # user_id should NOT be an alias when username is present
        assert "123456789" not in alias_values


class TestExtractRelationshipTerm:
    """Tests for extract_relationship_term helper."""

    def test_extracts_wife(self):
        assert extract_relationship_term("Sarah is the user's wife") == "wife"

    def test_extracts_boss(self):
        assert extract_relationship_term("John is the boss") == "boss"

    def test_extracts_best_friend_over_friend(self):
        """Multi-word terms matched before single-word substrings."""
        assert extract_relationship_term("Alex is my best friend") == "best friend"

    def test_no_match(self):
        assert extract_relationship_term("Sarah likes hiking") is None

    def test_case_insensitive(self):
        assert extract_relationship_term("Sarah is my Wife") == "wife"
