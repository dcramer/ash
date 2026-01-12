"""Memory store for memories, people, and user profiles.

Note: Session and message storage has been moved to ash.sessions module.
This module now only handles SQLite-based storage for:
- Memories (with embeddings for semantic search)
- People (relationship tracking)
- User profiles
- Skill state
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ash.db.models import (
    Memory,
    Person,
    SkillState,
    UserProfile,
)


class MemoryStore:
    """Store and retrieve memories, people, and user profiles."""

    def __init__(self, session: AsyncSession):
        """Initialize memory store.

        Args:
            session: Database session.
        """
        self._session = session

    # Person operations

    async def create_person(
        self,
        owner_user_id: str,
        name: str,
        relationship: str | None = None,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Person:
        """Create a new person entity.

        Args:
            owner_user_id: User who owns this person relationship.
            name: Person's primary name.
            relationship: Relationship type (wife, boss, friend, etc.).
            aliases: Alternative names or references.
            metadata: Optional metadata.

        Returns:
            Created person.
        """
        person = Person(
            id=str(uuid.uuid4()),
            owner_user_id=owner_user_id,
            name=name,
            relation=relationship,
            aliases=aliases or [],
            metadata_=metadata,
        )
        self._session.add(person)
        await self._session.flush()
        return person

    async def get_person(self, person_id: str) -> Person | None:
        """Get person by ID.

        Args:
            person_id: Person ID.

        Returns:
            Person or None if not found.
        """
        stmt = select(Person).where(Person.id == person_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def find_person_by_reference(
        self,
        owner_user_id: str,
        reference: str,
    ) -> Person | None:
        """Find person by name, relationship, or alias.

        Args:
            owner_user_id: The user who owns this person reference.
            reference: Name like "Sarah", relationship like "wife", or alias.

        Returns:
            Person if found, None otherwise.
        """
        reference_lower = reference.lower().strip()

        # Remove common prefixes
        for prefix in ["my ", "the "]:
            if reference_lower.startswith(prefix):
                reference_lower = reference_lower[len(prefix) :]

        stmt = select(Person).where(Person.owner_user_id == owner_user_id)
        result = await self._session.execute(stmt)
        people = result.scalars().all()

        for person in people:
            # Check name
            if person.name.lower() == reference_lower:
                return person
            # Check relationship
            if person.relation and person.relation.lower() == reference_lower:
                return person
            # Check aliases
            if person.aliases:
                for alias in person.aliases:
                    if alias.lower() == reference_lower:
                        return person

        return None

    async def get_people_for_user(self, owner_user_id: str) -> list[Person]:
        """Get all people for a user.

        Args:
            owner_user_id: User ID.

        Returns:
            List of people.
        """
        stmt = (
            select(Person)
            .where(Person.owner_user_id == owner_user_id)
            .order_by(Person.name)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update_person(
        self,
        person_id: str,
        name: str | None = None,
        relationship: str | None = None,
        aliases: list[str] | None = None,
    ) -> Person | None:
        """Update person details.

        Args:
            person_id: Person ID.
            name: New name (or None to keep current).
            relationship: New relationship (or None to keep current).
            aliases: New aliases (or None to keep current).

        Returns:
            Updated person or None if not found.
        """
        person = await self.get_person(person_id)
        if not person:
            return None

        if name is not None:
            person.name = name
        if relationship is not None:
            person.relation = relationship
        if aliases is not None:
            person.aliases = aliases

        await self._session.flush()
        return person

    async def add_person_alias(self, person_id: str, alias: str) -> Person | None:
        """Add an alias to a person.

        Args:
            person_id: Person ID.
            alias: Alias to add.

        Returns:
            Updated person or None if not found.
        """
        person = await self.get_person(person_id)
        if not person:
            return None

        aliases = list(person.aliases or [])
        if alias.lower() not in [a.lower() for a in aliases]:
            aliases.append(alias)
            person.aliases = aliases
            await self._session.flush()

        return person

    # Memory operations

    async def add_memory(
        self,
        content: str,
        source: str | None = None,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        subject_person_ids: list[str] | None = None,
    ) -> Memory:
        """Add a memory entry.

        Memory scoping:
        - Personal: owner_user_id set, chat_id NULL - only visible to that user
        - Group: owner_user_id NULL, chat_id set - visible to everyone in that chat

        Args:
            content: Memory content.
            source: Source of memory.
            expires_at: When this memory expires.
            metadata: Optional metadata.
            owner_user_id: User who added this memory (NULL for group memories).
            chat_id: Chat this memory belongs to (NULL for personal memories).
            subject_person_ids: List of person IDs this memory is about.

        Returns:
            Created memory entry.

        Raises:
            ValueError: If any subject_person_ids don't exist in the database.
        """
        # Validate subject_person_ids exist
        if subject_person_ids:
            for person_id in subject_person_ids:
                person = await self.get_person(person_id)
                if not person:
                    raise ValueError(f"Invalid subject person ID: {person_id}")

        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            source=source,
            expires_at=expires_at,
            metadata_=metadata,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            subject_person_ids=subject_person_ids,
        )
        self._session.add(memory)
        await self._session.flush()
        return memory

    async def get_memories(
        self,
        limit: int = 100,
        include_expired: bool = False,
        include_superseded: bool = False,
    ) -> list[Memory]:
        """Get memory entries.

        Args:
            limit: Maximum number of entries.
            include_expired: Include expired entries.
            include_superseded: Include superseded entries.

        Returns:
            List of memory entries.
        """
        stmt = select(Memory).order_by(Memory.created_at.desc()).limit(limit)

        if not include_expired:
            now = datetime.now(UTC)
            stmt = stmt.where((Memory.expires_at.is_(None)) | (Memory.expires_at > now))

        if not include_superseded:
            stmt = stmt.where(Memory.superseded_at.is_(None))

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_memories_about_person(
        self,
        person_id: str,
        limit: int = 50,
        include_expired: bool = False,
        include_superseded: bool = False,
    ) -> list[Memory]:
        """Get memory entries about a specific person.

        Args:
            person_id: Person ID.
            limit: Maximum number of entries.
            include_expired: Include expired entries.
            include_superseded: Include superseded entries.

        Returns:
            List of memory entries about this person.
        """
        from sqlalchemy import text

        # Use SQLite JSON function to check if person_id is in the array
        # json_each unpacks the array so we can search for the value
        stmt = (
            select(Memory)
            .where(
                text(
                    "EXISTS (SELECT 1 FROM json_each(memories.subject_person_ids) "
                    "WHERE json_each.value = :person_id)"
                ).bindparams(person_id=person_id)
            )
            .order_by(Memory.created_at.desc())
            .limit(limit)
        )

        if not include_expired:
            now = datetime.now(UTC)
            stmt = stmt.where((Memory.expires_at.is_(None)) | (Memory.expires_at > now))

        if not include_superseded:
            stmt = stmt.where(Memory.superseded_at.is_(None))

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def mark_memory_superseded(
        self,
        memory_id: str,
        superseded_by_id: str,
    ) -> bool:
        """Mark a memory as superseded by another memory.

        Args:
            memory_id: ID of the memory to mark as superseded.
            superseded_by_id: ID of the newer memory that supersedes this one.

        Returns:
            True if updated, False if memory not found.
        """
        stmt = select(Memory).where(Memory.id == memory_id)
        result = await self._session.execute(stmt)
        memory = result.scalar_one_or_none()

        if not memory:
            return False

        memory.superseded_at = datetime.now(UTC)
        memory.superseded_by_id = superseded_by_id
        await self._session.flush()
        return True

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Get memory by ID.

        Args:
            memory_id: Memory ID.

        Returns:
            Memory or None if not found.
        """
        stmt = select(Memory).where(Memory.id == memory_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    # User profile operations

    async def get_or_create_user_profile(
        self,
        user_id: str,
        provider: str,
        username: str | None = None,
        display_name: str | None = None,
    ) -> UserProfile:
        """Get or create user profile.

        Args:
            user_id: User ID from provider.
            provider: Provider name.
            username: Username.
            display_name: Display name.

        Returns:
            User profile.
        """
        stmt = select(UserProfile).where(UserProfile.user_id == user_id)
        result = await self._session.execute(stmt)
        profile = result.scalar_one_or_none()

        if profile is None:
            profile = UserProfile(
                user_id=user_id,
                provider=provider,
                username=username,
                display_name=display_name,
            )
            self._session.add(profile)
            await self._session.flush()
        else:
            # Update if new info provided
            if username and profile.username != username:
                profile.username = username
            if display_name and profile.display_name != display_name:
                profile.display_name = display_name
            await self._session.flush()

        return profile

    # Skill state operations

    async def get_skill_state(
        self,
        skill_name: str,
        key: str,
        user_id: str | None = None,
    ) -> Any | None:
        """Get a skill state value.

        Args:
            skill_name: Name of the skill.
            key: State key.
            user_id: User ID for user-scoped state (None for global).

        Returns:
            State value or None if not found.
        """
        stmt = select(SkillState).where(
            SkillState.skill_name == skill_name,
            SkillState.key == key,
            SkillState.user_id == (user_id or ""),
        )
        result = await self._session.execute(stmt)
        state = result.scalar_one_or_none()
        return state.value if state else None

    async def set_skill_state(
        self,
        skill_name: str,
        key: str,
        value: Any,
        user_id: str | None = None,
    ) -> SkillState:
        """Set a skill state value.

        Args:
            skill_name: Name of the skill.
            key: State key.
            value: State value (will be serialized as JSON).
            user_id: User ID for user-scoped state (None for global).

        Returns:
            Created or updated skill state.
        """
        user_id_val = user_id or ""

        stmt = select(SkillState).where(
            SkillState.skill_name == skill_name,
            SkillState.key == key,
            SkillState.user_id == user_id_val,
        )
        result = await self._session.execute(stmt)
        state = result.scalar_one_or_none()

        if state is None:
            state = SkillState(
                skill_name=skill_name,
                key=key,
                user_id=user_id_val,
                value=value,
            )
            self._session.add(state)
        else:
            state.value = value

        await self._session.flush()
        return state

    async def delete_skill_state(
        self,
        skill_name: str,
        key: str,
        user_id: str | None = None,
    ) -> bool:
        """Delete a skill state value.

        Args:
            skill_name: Name of the skill.
            key: State key.
            user_id: User ID for user-scoped state (None for global).

        Returns:
            True if deleted, False if not found.
        """
        stmt = select(SkillState).where(
            SkillState.skill_name == skill_name,
            SkillState.key == key,
            SkillState.user_id == (user_id or ""),
        )
        result = await self._session.execute(stmt)
        state = result.scalar_one_or_none()

        if state:
            await self._session.delete(state)
            await self._session.flush()
            return True
        return False

    async def get_all_skill_state(
        self,
        skill_name: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Get all state values for a skill.

        Args:
            skill_name: Name of the skill.
            user_id: User ID for user-scoped state (None for global).

        Returns:
            Dict mapping keys to values.
        """
        stmt = select(SkillState).where(
            SkillState.skill_name == skill_name,
            SkillState.user_id == (user_id or ""),
        )
        result = await self._session.execute(stmt)
        states = result.scalars().all()
        return {state.key: state.value for state in states}
