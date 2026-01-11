"""Memory store for conversation history and knowledge."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ash.db.models import Knowledge, Message, Session, ToolExecution, UserProfile


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

    # Knowledge operations

    async def add_knowledge(
        self,
        content: str,
        source: str | None = None,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Knowledge:
        """Add knowledge to the knowledge base.

        Args:
            content: Knowledge content.
            source: Source of knowledge.
            expires_at: When this knowledge expires.
            metadata: Optional metadata.

        Returns:
            Created knowledge entry.
        """
        knowledge = Knowledge(
            id=str(uuid.uuid4()),
            content=content,
            source=source,
            expires_at=expires_at,
            metadata_=metadata,
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
