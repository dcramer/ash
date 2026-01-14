"""Memory store for memories, people, and user profiles."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ash.db.models import (
    Memory,
    Person,
    UserProfile,
)


class MemoryStore:
    """Store and retrieve memories, people, and user profiles."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_person(
        self,
        owner_user_id: str,
        name: str,
        relationship: str | None = None,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Person:
        """Create a new person entity."""
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

    async def get_person(
        self,
        person_id: str,
        owner_user_id: str | None = None,
    ) -> Person | None:
        """Get person by ID, optionally filtering by owner."""
        result = await self._session.execute(
            select(Person).where(Person.id == person_id)
        )
        person = result.scalar_one_or_none()

        if person and owner_user_id and person.owner_user_id != owner_user_id:
            return None

        return person

    async def find_person_by_reference(
        self,
        owner_user_id: str,
        reference: str,
    ) -> Person | None:
        """Find person by name, relationship, or alias."""
        ref = reference.lower().strip()
        for prefix in ["my ", "the ", "@"]:
            if ref.startswith(prefix):
                ref = ref[len(prefix) :]

        result = await self._session.execute(
            select(Person).where(Person.owner_user_id == owner_user_id)
        )

        for person in result.scalars().all():
            if person.name.lower() == ref:
                return person
            if person.relation and person.relation.lower() == ref:
                return person
            if person.aliases:
                for alias in person.aliases:
                    if alias.lower() == ref:
                        return person

        return None

    async def get_people_for_user(self, owner_user_id: str) -> list[Person]:
        """Get all people for a user."""
        result = await self._session.execute(
            select(Person)
            .where(Person.owner_user_id == owner_user_id)
            .order_by(Person.name)
        )
        return list(result.scalars().all())

    async def update_person(
        self,
        person_id: str,
        owner_user_id: str,
        name: str | None = None,
        relationship: str | None = None,
        aliases: list[str] | None = None,
    ) -> Person | None:
        """Update person details."""
        person = await self.get_person(person_id, owner_user_id=owner_user_id)
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

    async def add_person_alias(
        self,
        person_id: str,
        alias: str,
        owner_user_id: str,
    ) -> Person | None:
        """Add an alias to a person."""
        person = await self.get_person(person_id, owner_user_id=owner_user_id)
        if not person:
            return None

        aliases = list(person.aliases or [])
        if alias.lower() not in [a.lower() for a in aliases]:
            aliases.append(alias)
            person.aliases = aliases
            await self._session.flush()

        return person

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
        """Add a memory entry."""
        if subject_person_ids:
            for person_id in subject_person_ids:
                # Verify person exists and belongs to the same owner
                person = await self.get_person(person_id, owner_user_id=owner_user_id)
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
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[Memory]:
        """Get memory entries."""
        from sqlalchemy import or_

        stmt = select(Memory).order_by(Memory.created_at.desc()).limit(limit)

        if not include_expired:
            now = datetime.now(UTC)
            stmt = stmt.where((Memory.expires_at.is_(None)) | (Memory.expires_at > now))

        if not include_superseded:
            stmt = stmt.where(Memory.superseded_at.is_(None))

        if owner_user_id or chat_id:
            conditions = []
            if owner_user_id:
                conditions.append(Memory.owner_user_id == owner_user_id)
            if chat_id:
                conditions.append(
                    (Memory.chat_id == chat_id) & (Memory.owner_user_id.is_(None))
                )
            stmt = stmt.where(or_(*conditions))

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_memories_about_person(
        self,
        person_id: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
        limit: int = 50,
        include_expired: bool = False,
        include_superseded: bool = False,
    ) -> list[Memory]:
        """Get memory entries about a specific person."""
        from sqlalchemy import or_, text

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

        if owner_user_id or chat_id:
            conditions = []
            if owner_user_id:
                conditions.append(Memory.owner_user_id == owner_user_id)
            if chat_id:
                conditions.append(
                    (Memory.chat_id == chat_id) & (Memory.owner_user_id.is_(None))
                )
            stmt = stmt.where(or_(*conditions))

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def mark_memory_superseded(
        self,
        memory_id: str,
        superseded_by_id: str,
    ) -> bool:
        """Mark a memory as superseded by another memory."""
        result = await self._session.execute(
            select(Memory).where(Memory.id == memory_id)
        )
        memory = result.scalar_one_or_none()
        if not memory:
            return False

        memory.superseded_at = datetime.now(UTC)
        memory.superseded_by_id = superseded_by_id
        await self._session.flush()
        return True

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Get memory by ID."""
        result = await self._session.execute(
            select(Memory).where(Memory.id == memory_id)
        )
        return result.scalar_one_or_none()

    async def delete_memory(
        self,
        memory_id: str,
        owner_user_id: str | None = None,
        chat_id: str | None = None,
    ) -> bool:
        """Delete a memory by ID, optionally verifying ownership."""
        memory = await self.get_memory(memory_id)
        if not memory:
            return False

        if owner_user_id or chat_id:
            is_owner = (
                memory.owner_user_id == owner_user_id and owner_user_id is not None
            )
            is_group_member = memory.owner_user_id is None and memory.chat_id == chat_id
            if not (is_owner or is_group_member):
                return False

        await self._session.delete(memory)
        await self._session.flush()
        return True

    async def get_or_create_user_profile(
        self,
        user_id: str,
        provider: str,
        username: str | None = None,
        display_name: str | None = None,
    ) -> UserProfile:
        """Get or create user profile."""
        result = await self._session.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
        profile = result.scalar_one_or_none()

        if profile is None:
            profile = UserProfile(
                user_id=user_id,
                provider=provider,
                username=username,
                display_name=display_name,
            )
            self._session.add(profile)
        else:
            if username and profile.username != username:
                profile.username = username
            if display_name and profile.display_name != display_name:
                profile.display_name = display_name

        await self._session.flush()
        return profile
