"""Memory store for conversation history and knowledge."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ash.db.models import (
    Knowledge,
    Message,
    Person,
    Session,
    SkillState,
    ToolExecution,
    UserProfile,
)


class MemoryStore:
    """Store and retrieve conversation history and knowledge."""

    def __init__(self, session: AsyncSession):
        """Initialize memory store.

        Args:
            session: Database session.
        """
        self._session = session

    # Session operations

    async def get_or_create_session(
        self,
        provider: str,
        chat_id: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Get existing session or create a new one.

        Args:
            provider: Provider name (e.g., 'telegram').
            chat_id: Chat identifier from provider.
            user_id: User identifier from provider.
            metadata: Optional session metadata.

        Returns:
            Session instance.
        """
        stmt = select(Session).where(
            Session.provider == provider,
            Session.chat_id == chat_id,
        )
        result = await self._session.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            session = Session(
                id=str(uuid.uuid4()),
                provider=provider,
                chat_id=chat_id,
                user_id=user_id,
                metadata_=metadata,
            )
            self._session.add(session)
            await self._session.flush()

        return session

    async def get_session(self, session_id: str) -> Session | None:
        """Get session by ID.

        Args:
            session_id: Session ID.

        Returns:
            Session or None if not found.
        """
        stmt = select(Session).where(Session.id == session_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    # Message operations

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        token_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Add a message to session history.

        Args:
            session_id: Session ID.
            role: Message role (user, assistant, system).
            content: Message content.
            token_count: Optional token count.
            metadata: Optional message metadata.

        Returns:
            Created message.
        """
        message = Message(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            token_count=token_count,
            metadata_=metadata,
        )
        self._session.add(message)
        await self._session.flush()
        return message

    async def get_messages(
        self,
        session_id: str,
        limit: int = 50,
        before: datetime | None = None,
    ) -> list[Message]:
        """Get messages for a session.

        Args:
            session_id: Session ID.
            limit: Maximum number of messages.
            before: Only get messages before this time.

        Returns:
            List of messages, oldest first.
        """
        stmt = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )

        if before:
            stmt = stmt.where(Message.created_at < before)

        result = await self._session.execute(stmt)
        messages = list(result.scalars().all())
        messages.reverse()  # Return oldest first
        return messages

    async def has_message_with_external_id(
        self,
        session_id: str,
        external_id: str,
    ) -> bool:
        """Check if a message with given external ID exists.

        Used to avoid processing duplicate messages (e.g., from Telegram).

        Args:
            session_id: Session ID.
            external_id: External message ID (e.g., Telegram message ID).

        Returns:
            True if message exists, False otherwise.
        """
        from sqlalchemy import cast, func
        from sqlalchemy.dialects.sqlite import JSON

        # Check if any message in this session has this external_id in metadata
        stmt = select(Message).where(
            Message.session_id == session_id,
            Message.role == "user",
            func.json_extract(Message.metadata_, "$.external_id") == external_id,
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None

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
            relationship=relationship,
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

    # Knowledge operations

    async def add_knowledge(
        self,
        content: str,
        source: str | None = None,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        owner_user_id: str | None = None,
        subject_person_id: str | None = None,
    ) -> Knowledge:
        """Add knowledge to the knowledge base.

        Args:
            content: Knowledge content.
            source: Source of knowledge.
            expires_at: When this knowledge expires.
            metadata: Optional metadata.
            owner_user_id: User who added this knowledge.
            subject_person_id: Person this knowledge is about.

        Returns:
            Created knowledge entry.
        """
        knowledge = Knowledge(
            id=str(uuid.uuid4()),
            content=content,
            source=source,
            expires_at=expires_at,
            metadata_=metadata,
            owner_user_id=owner_user_id,
            subject_person_id=subject_person_id,
        )
        self._session.add(knowledge)
        await self._session.flush()
        return knowledge

    async def get_knowledge(
        self,
        limit: int = 100,
        include_expired: bool = False,
    ) -> list[Knowledge]:
        """Get knowledge entries.

        Args:
            limit: Maximum number of entries.
            include_expired: Include expired entries.

        Returns:
            List of knowledge entries.
        """
        stmt = select(Knowledge).order_by(Knowledge.created_at.desc()).limit(limit)

        if not include_expired:
            now = datetime.now(UTC)
            stmt = stmt.where(
                (Knowledge.expires_at.is_(None)) | (Knowledge.expires_at > now)
            )

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_knowledge_about_person(
        self,
        person_id: str,
        limit: int = 50,
        include_expired: bool = False,
    ) -> list[Knowledge]:
        """Get knowledge entries about a specific person.

        Args:
            person_id: Person ID.
            limit: Maximum number of entries.
            include_expired: Include expired entries.

        Returns:
            List of knowledge entries about this person.
        """
        stmt = (
            select(Knowledge)
            .where(Knowledge.subject_person_id == person_id)
            .order_by(Knowledge.created_at.desc())
            .limit(limit)
        )

        if not include_expired:
            now = datetime.now(UTC)
            stmt = stmt.where(
                (Knowledge.expires_at.is_(None)) | (Knowledge.expires_at > now)
            )

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

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

    # Tool execution operations

    async def log_tool_execution(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        output: str | None,
        success: bool,
        duration_ms: int | None = None,
        session_id: str | None = None,
    ) -> ToolExecution:
        """Log a tool execution.

        Args:
            tool_name: Name of the tool.
            input_data: Tool input.
            output: Tool output.
            success: Whether execution succeeded.
            duration_ms: Execution duration in milliseconds.
            session_id: Optional associated session.

        Returns:
            Created tool execution record.
        """
        execution = ToolExecution(
            id=str(uuid.uuid4()),
            session_id=session_id,
            tool_name=tool_name,
            input=input_data,
            output=output,
            success=success,
            duration_ms=duration_ms,
        )
        self._session.add(execution)
        await self._session.flush()
        return execution

    async def get_tool_executions(
        self,
        session_id: str | None = None,
        tool_name: str | None = None,
        limit: int = 50,
    ) -> list[ToolExecution]:
        """Get tool execution history.

        Args:
            session_id: Filter by session.
            tool_name: Filter by tool name.
            limit: Maximum number of records.

        Returns:
            List of tool executions.
        """
        stmt = (
            select(ToolExecution).order_by(ToolExecution.created_at.desc()).limit(limit)
        )

        if session_id:
            stmt = stmt.where(ToolExecution.session_id == session_id)
        if tool_name:
            stmt = stmt.where(ToolExecution.tool_name == tool_name)

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

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
